import sys

import torch
from PIL import Image
from torchvision import utils

from simple_ddpm.gaussian_diffusion import GaussianDiffusion
from simple_ddpm.trainer import DEVICE
from simple_ddpm.unet import Unet
from simple_ddpm.util import ImageTransform


def inpaint(model_file: str, image_file: str, masking_file: str, image_size: int = 128):
    denoise_model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).to(DEVICE)

    diffusion_model = GaussianDiffusion(
        denoise_model,
        n_steps=200,
        image_size=image_size,
        channels=3,
        device=DEVICE
    ).to(DEVICE)

    diffusion_model.load_state_dict(torch.load(model_file, map_location=DEVICE)['ema'])

    from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, \
        CenterCrop, ToTensor, Lambda

    # image_transform = ImageTransform(image_size=128)

    image_transform = Compose([
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),
            Lambda(lambda t: (t * 2) - 1)
        ])
    image = image_transform(Image.open(image_file))
    image = image.to(DEVICE)

    mask_transform = Compose([
            Resize(image_size),
            CenterCrop(image_size),
            ToTensor(),
            Lambda(lambda t: 1 - t)
        ])
    mask = Image.open(masking_file)
    mask = mask_transform(mask)
    mask = mask.to(DEVICE)

    # Manual masking
    mask = torch.ones_like(image)
    mask[:, :, 40:] = 0    

    # DEBUG ONLY
    input_image = image * mask
    utils.save_image(input_image, "inpaint-input.png")

    repaint_steps = 1  # DEBUG: no resampling
    output_image = diffusion_model.inpaint(image, mask, repaint_steps=repaint_steps, batch_size=7)
    output_image = (output_image + 1) * 0.5
    utils.save_image(output_image, 'inpaint-output.png')


def main():
    model_file = sys.argv[1]
    image_file = sys.argv[2]
    masking_file = sys.argv[3]
    inpaint(model_file, image_file, masking_file, image_size=64)
    print("Finished")

if __name__ == "__main__":
    main()
