from datetime import datetime
import os
import time
import wandb
import argparse
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from os.path import join

# Set CUDA architecture list   
from dose_plus.util.other import set_torch_cuda_arch_list
import torch
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')


from dose_plus.backbones.shared import BackboneRegistry
from dose_plus.data_module import SpecsDataModule
from dose_plus.sdes import SDERegistry
from dose_plus.model import ScoreModel

import json

def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups

if __name__ == '__main__':
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--config_path", type=str, help="Path to config file.")
          tmp_args = parser_.parse_args() 
          config_path=tmp_args.config_path
          with open(config_path) as f:
               config = json.load(f)

          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default='ncsnpp')
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default='dose_plus')
          parser_.add_argument("--nolog", default=True,action='store_true', help="Turn off logging.")
          parser_.add_argument("--wandb_name", type=str, default=None, help="Name for wandb logger. If not set, a random name is generated.")
          parser_.add_argument("--log_dir",type=str, help="Directory to save logs.")
          parser_.add_argument("--valid_only", default=False, help="Whether to only validate the model.")
          parser_.add_argument("--best_ckpt", help="Whether to only validate the model.")
     
     temp_args, _ = base_parser.parse_known_args()

     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
     trainer_parser.add_argument("--accelerator", type=str, default="gpu", help="Supports passing different accelerator types.")
     trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients.")
     trainer_parser.add_argument("--max_epochs", type=int, default=-1, help="Number of epochs to train.")
     trainer_parser.add_argument("--devices", default=[0], help="How many gpus to use.")
     trainer_parser.add_argument("--num_nodes", default=1, help="Number of nodes(devices).")
     tr_args = parser.parse_args()
     if hasattr(tr_args, 'num_nodes'):
          if tr_args.num_nodes > 1:
               mul_dev=True
          else:
               mul_dev=False
     else:
          mul_dev=False

     ScoreModel.add_argparse_args(
          parser.add_argument_group("ScoreModel", description=ScoreModel.__name__),
          config
          )
     ScoreModel.valid_only = tr_args.valid_only

     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
     
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__),
          config
          )
     data_module_cls.ddp=mul_dev   

     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)

     # Initialize logger, trainer, model, datamodule
     model = ScoreModel(
          backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
          **{
               **vars(arg_groups['ScoreModel']),
               **vars(arg_groups['SDE']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule'])
          }
     )

     # Set up logger configuration
     if args.nolog:
          logdir = args.log_dir
          logger = TensorBoardLogger(save_dir=logdir, name="tensorboard", version="1")
     else:
          logger = WandbLogger(project="xxx", log_model=True, save_dir="logs", name=args.wandb_name)
          logger.experiment.log_code(".")

     # Set up callbacks for logger
     if logger != None:
          callbacks = [
               ModelCheckpoint(dirpath=join(args.log_dir), save_last=True,filename='{epoch}'),
               ModelCheckpoint(dirpath=join(args.log_dir), save_top_k=2, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}'),

               ]
     else:
          callbacks = None 

     # Initialize the Trainer and the DataModule
     percent_data=1.0 if not args.valid_only else 0.001
     trainer = pl.Trainer(
          **vars(arg_groups['Trainer']),
          strategy="ddp", logger=logger,
          log_every_n_steps=10, num_sanity_val_steps=0,
          callbacks=callbacks,
          limit_train_batches=percent_data 
     )
     
     # Train model
     ckpt_path = tr_args.best_ckpt if os.path.exists(tr_args.best_ckpt) else args.log_dir+'/last.ckpt'
     print(ckpt_path)
     if os.path.exists(ckpt_path):
          trainer.fit(model, ckpt_path=ckpt_path)
     else:
          print("No checkpoint found, training from 0.")
          trainer.fit(model)