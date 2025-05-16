# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
from scipy import integrate
import torch

from .predictors import Predictor, PredictorRegistry, ReverseDiffusionPredictor
from .correctors import Corrector, CorrectorRegistry

import matplotlib.pyplot as plt
import numpy as np
from soundfile import write


__all__ = [
    'PredictorRegistry', 'CorrectorRegistry', 'Predictor', 'Corrector',
    'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_sampler(
    predictor_name, corrector_name, sde, score_fn, y, denoise=True, eps=4e-2, 
    s1=35, s2=10, snr=0.1, corrector_steps=1, probability_flow: bool = False,**kwargs
):
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow,**kwargs)
    corrector = corrector_cls(sde, score_fn, snr=snr, n_steps=corrector_steps, **kwargs)

    def sampler():
        with torch.no_grad():
            timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
            steps=[sde.N-s1,sde.N-s2]
            t_start= timesteps[steps[0]]
            mean_t_start, std_t_start = sde.marginal_prob(y, t_start, y)
            if std_t_start.dim() == 0:
                std_t_start = std_t_start.unsqueeze(0)
            sigmas_t_start = std_t_start[:, None, None, None]
            z_t_start = torch.randn_like(y)
            xt = mean_t_start + sigmas_t_start * z_t_start 
            for i,(step) in enumerate(steps):
                t=timesteps[step]
                vec_t = torch.ones(y.shape[0], device=y.device) * t
                xt, xt_mean = predictor.update_fn(xt, vec_t, y)
            x_result = xt_mean if denoise else xt 
            ns = sde.N * (corrector.n_steps + 1)
            return x_result, ns
    return sampler
