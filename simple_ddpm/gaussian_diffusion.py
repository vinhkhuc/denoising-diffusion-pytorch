"""
Paper: "Denoising Diffusion Probabilistic Models", https://arxiv.org/abs/2006.11239
Code adapted from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/ddpm
"""
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from simple_ddpm.util import gather


class GaussianDiffusion(nn.Module):
    def __init__(self, eps_model: nn.Module, n_steps: int, image_size: int, channels: int, device: torch.device):
        super().__init__()
        self.eps_model = eps_model
        self.image_size = image_size
        self.channels = channels
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    # FIXME: There must be no noise when t = 0
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps_theta = self.eps_model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** 0.5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t)
        return F.mse_loss(noise, eps_theta)

    def forward(self, x0, *args, **kwargs) -> torch.Tensor:
        return self.loss(x0, *args, **kwargs)

    @torch.no_grad()
    def p_sample_loop(self, shape) -> torch.Tensor:
        device = self.beta.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.n_steps)), desc='sampling loop time step', total=self.n_steps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def sample(self, batch_size: int = 16) -> torch.Tensor:
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def inpaint_sample(self, xt: torch.Tensor, x0: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # noise_known and noise_unknown are drawn inside q_sample() and p_sample
        x_tm1_known = self.q_sample(x0, t)
        x_tm1_unknown = self.p_sample(xt, t)

        return mask * x_tm1_known + (1 - mask) * x_tm1_unknown

    def q_xt_xtm1(self, x_tm1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t-1) ** 0.5 * x_tm1
        var = 1 - gather(self.alpha_bar, t-1)
        return mean, var

    # TODO: Rewrite the loop
    # TODO: Move repaint_steps to the class
    # TODO: No need to putting torch.no_grad() everywhere
    @torch.no_grad()
    def inpaint_sample_loop(self, x0, mask, repaint_steps, shape) -> torch.Tensor:
        device = self.beta.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.n_steps)), desc='sampling loop time step', total=self.n_steps):
            for u in tqdm(reversed(range(0, repaint_steps)), desc='re-sampling loop', total=repaint_steps):
                t = torch.full((b,), i, device=device, dtype=torch.long)
                img = self.inpaint_sample(img, x0, mask, t)

                # DEBUG: no re-sampling
                # if u < repaint_steps and i > 1:
                #     mean, var = self.q_xt_xtm1(img, t)
                #     img = mean + var ** 0.5

        return img

    @torch.no_grad()
    def inpaint(self, x0: torch.Tensor, mask: torch.Tensor, repaint_steps=2, batch_size: int = 16) -> torch.Tensor:
        image_size = self.image_size
        channels = self.channels
        return self.inpaint_sample_loop(x0, mask, repaint_steps, shape=(batch_size, channels, image_size, image_size))
