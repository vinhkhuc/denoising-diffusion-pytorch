"""
Adapted from: https://github.com/lucidrains/denoising-diffusion-pytorch
"""
from inspect import isfunction

import torch
from torch import Tensor
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, \
    CenterCrop, ToTensor, Lambda

from PIL import Image


try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageTransform:
    def __init__(self, image_size: int):
        self.image_size = image_size

        self.transform = Compose([
            Resize(image_size),
            RandomHorizontalFlip(),
            CenterCrop(image_size),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)
        ])

    def __call__(self, img: Image) -> Tensor:
        return self.transform(img)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def gather(consts: torch.Tensor, t: torch.Tensor):
    """
    Gather consts for $t$ and reshape to feature map shape
    """
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


# def extract(a, t, x_shape):
#     b, *_ = t.shape
#     out = a.gather(-1, t)
#     return out.reshape(b, *((1,) * (len(x_shape) - 1)))
#
#
# def noise_like(shape, device, repeat=False):
#     repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
#     noise = lambda: torch.randn(shape, device=device)
#     return repeat_noise() if repeat else noise()
#
#
# def cosine_beta_schedule(timesteps, s=0.008):
#     """
#     cosine schedule
#     as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
#     """
#     steps = timesteps + 1
#     x = np.linspace(0, steps, steps)
#     alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return np.clip(betas, a_min=0, a_max=0.999)
