import time
from math import ceil
import warnings

from matplotlib import pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage

from dose_plus import sampling
from dose_plus.sdes import SDERegistry
from dose_plus.backbones import BackboneRegistry
from dose_plus.util.inference import evaluate_model
from dose_plus.util.other import pad_spec

import json
import os
import torch.nn.functional as F

class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser,config):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.04, help="The minimum process time (0.04 by default)")
        parser.add_argument("--num_eval_files", type=int, default=1, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", choices=("mse", "mae"), help="The type of loss function to use.")
        parser.add_argument("--rt", default='t', help="dropout rate.")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=0.03,
        num_eval_files=1, loss_type='mse', data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files

        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self.rt = kwargs.get('rt', 0.0)
        self.sde_name=sde

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _loss(self, err):
        if self.loss_type == 'mse':
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))

        return loss

    def _step(self, batch, batch_idx,mode='valid'):
        x, y = batch
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  
        sigmas = std[:, None, None, None]
        x_t = mean + sigmas * z

        if mode=='train':
            if self.rt =="t":
                # time-aware dropout
                self.rt=(self.dnn.get_rt(t))
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.rt)))
                logits = torch.stack([self.rt, 1 - self.rt], dim=1) + gumbel_noise.unsqueeze(1)
                temperature = 1 
                self.masks = F.gumbel_softmax(logits, tau=temperature, hard=True)[:, 0].view(-1, 1, 1, 1)
                noise = torch.randn_like(x_t)
                x_t = x_t * (1 - self.masks) + noise * self.masks
                
            elif self.rt > 0:
                masks = torch.bernoulli(torch.zeros(x.shape[0]) + self.rt)
                for i in range(masks.size(0)):
                    if masks[i]:
                        x_t[i] = torch.randn_like(x_t[i])

        hat_x=self.dnn(torch.cat([x_t, y], dim=1), t)
        err = hat_x - x
        loss = self._loss(err)

        return loss if not self.valid_only else 0*loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx,mode='train')
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)    
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        
        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, estoi = evaluate_model(self, self.num_eval_files, 
                sde_type=self.sde_name,rt=self.rt)
            
            print(f"Validation performance: PESQ={pesq:.2f}, ESTOI={estoi:.2f}")
            if self.valid_only:
                exit()
            self.log('pesq', pesq, on_step=False, on_epoch=True, sync_dist=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None,
                       X=None, wav_dict=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps,"sde_name":self.sde_name, **kwargs}
        if minibatch is None:
            return sampling.get_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn
        
    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)