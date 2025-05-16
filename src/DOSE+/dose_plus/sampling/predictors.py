import abc

import torch
import numpy as np

from dose_plus.util.registry import Registry


PredictorRegistry = Registry("Predictor")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False,**kwargs):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn
        self.probability_flow = probability_flow

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

    def debug_update_fn(self, x, t, *args):
        raise NotImplementedError(f"Debug update function not implemented for predictor {self}.")
    

@PredictorRegistry.register('reverse_dose_plus')
class ReverseDDPMPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False,**kwargs):
        super().__init__(sde, score_fn, probability_flow=probability_flow,**kwargs)
        self.score_fn = score_fn

    def update_fn(self, xt1, t1, y):
        x0 = self.score_fn.dnn(torch.cat([xt1,y], dim=1),t1)
        z = torch.randn_like(xt1) 
        mean, std = self.sde.reverse(x0, xt1, t1)
        xt2 = mean + std[:, None, None, None]* z
        return xt2, x0
