import sys


from simple_ddpm.gaussian_diffusion import GaussianDiffusion
from simple_ddpm.trainer import Trainer, DEVICE
from simple_ddpm.unet import Unet


def train(image_dir: str):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).to(DEVICE)

    diffusion = GaussianDiffusion(
        model,
        n_steps=1000,
        image_size=128,
        channels=3,
        device=DEVICE
    ).to(DEVICE)

    Trainer(
        diffusion,
        image_dir,
        train_batch_size=32,
        train_lr=2e-5,
        train_num_steps=700000,
        save_and_sample_every=1000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        fp16=False
    ).train()


def main():
    image_dir = sys.argv[1]
    train(image_dir)


if __name__ == "__main__":
    main()
