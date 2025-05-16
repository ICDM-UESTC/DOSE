import abc
import warnings

import numpy as np
from dose_plus.util.tensors import batch_broadcast
import torch

from dose_plus.util.registry import Registry
import scipy.special as sc

SDERegistry = Registry("SDE")


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass

    def discretize(self, x, t, y, stepsize):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = stepsize
        drift, diffusion = self.sde(x, t, y)
        f = drift * dt
        G = diffusion * torch.sqrt(dt)
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, t, *args)
                total_drift, diffusion = rsde_parts["total_drift"], rsde_parts["diffusion"]
                return total_drift, diffusion

            def rsde_parts(self, x, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, t, *args)
                score = score_model(x, t, *args)
                score_drift = -sde_diffusion[:, None, None, None]**2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(sde_diffusion) if self.probability_flow else sde_diffusion
                total_drift = sde_drift + score_drift
                return {
                    'total_drift': total_drift, 'diffusion': diffusion, 'sde_drift': sde_drift,
                    'sde_diffusion': sde_diffusion, 'score_drift': score_drift, 'score': score,
                }

            def discretize(self, x, t, y, stepsize):
                f, G = discretize_fn(x, t, y, stepsize)
                score=score_model(x, t, y)
                rev_f = f - G[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G, score

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass

 
@SDERegistry.register("dose_plus")
class DOSE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--N", type=int, default=50,
            help="The number of timesteps in the SDE discretization. 1000 by default")
        parser.add_argument("--beta-min", type=float, default=0.5, help="The minimum beta to use.")
        parser.add_argument("--beta-max", type=float, default=10, help="The maximum beta to use.")

        return parser

    def __init__(self, beta_min, beta_max, N=50, **ignored_kwargs):
        super().__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        self.t = 1.

        self.betas = np.linspace(self.beta_min / self.N, self.beta_max / self.N, self.N)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = np.sqrt(1 - self.alphas_cumprod)

    def copy(self):
        return DOSE(self.beta_min, self.beta_max, N=self.N)

    @property
    def T(self):
        return self.t

    def _beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def integral_beta(self, t):
        return self.beta_min*t + .5*t**2*(self.beta_max - self.beta_min)

    def sde(self, x, t, y):
        beta = self._beta(t)[:, None, None, None]
        drift = -0.5 *beta*x
        diffusion = torch.sqrt(beta).squeeze(-1).squeeze(-1).squeeze(-1)
        return drift, diffusion
    

    def _mean(self, x0, t, y):
        integral_beta = self.integral_beta(t)
        if integral_beta.dim() == 0:
            integral_beta = integral_beta.unsqueeze(0)
        integral_beta=integral_beta[:, None, None, None]
        exp_itg_beta = torch.exp(-.5*integral_beta)
        return exp_itg_beta*x0

    def _std(self, t):
        integral_beta = self.integral_beta(t)
        exp_itg_beta = torch.exp(-integral_beta)
        if exp_itg_beta.dim() == 0:
            exp_itg_beta = exp_itg_beta.unsqueeze(0)
        return torch.sqrt(1 - exp_itg_beta)

    def marginal_prob(self, x0, t, y):
        mean = self._mean(x0, t, y)
        std = self._std(t)
        return mean, std

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        t = self.T*torch.ones(y.shape[0], device=y.device)
        mean, std = self.marginal_prob(y, t, y)
        x_T = mean + torch.randn_like(y, device=y.device) * std
        return x_T
    
    def reverse(self,x0,xt,t):
        timestep = (t * (self.N - 1) / self.T).type(torch.int)
        betas=torch.tensor(self.betas,device=x0.device)
        alphas=torch.tensor(self.alphas,device=x0.device)
        alphas_cumprod=torch.tensor(self.alphas_cumprod,device=x0.device)

        if timestep > 0:
            posterior_variance = (
                betas[timestep] 
                * (1.0 - alphas_cumprod[timestep-1]) / (1.0 - alphas_cumprod[timestep])
            )
            posterior_mean_coef1 = (
                betas[timestep] * torch.sqrt(alphas_cumprod[timestep-1]) / (1.0 - alphas_cumprod[timestep])
            )
            posterior_mean_coef2 = (
                (1.0 - alphas_cumprod[timestep-1])* 
                torch.sqrt(alphas[timestep])/
                 (1.0 - alphas_cumprod[timestep])
            )
            mean=posterior_mean_coef1[:, None, None, None]*x0+posterior_mean_coef2[:, None, None, None]*xt
            std=torch.sqrt(posterior_variance)
        else:
            mean=x0
            std=torch.zeros_like(betas[timestep])
        return mean.to(torch.complex64),std.to(torch.float32)