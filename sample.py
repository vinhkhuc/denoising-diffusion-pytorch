import sys

import torch
from torchvision import utils

from simple_ddpm.gaussian_diffusion import GaussianDiffusion
from simple_ddpm.trainer import DEVICE
from simple_ddpm.unet import Unet


def sample(model_file: str):
    denoise_model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).to(DEVICE)

    diffusion_model = GaussianDiffusion(
        denoise_model,
        n_steps=20,
        image_size=128,
        channels=3,
        device=DEVICE
    ).to(DEVICE)

    diffusion_model.load_state_dict(torch.load(model_file, map_location=DEVICE)['ema'])
    image = diffusion_model.sample(batch_size=7)
    image = (image + 1) * 0.5
    utils.save_image(image, 'sample.png')


def main():
    model_file = sys.argv[1]
    sample(model_file)
    print("Finished")

if __name__ == "__main__":
    main()
