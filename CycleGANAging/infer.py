import os
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import numpy as np
import math

from gan_module import Generator

parser = ArgumentParser()
parser.add_argument(
    '--image_dir', default='C:/Users/Admin/Documents/XU LY ANH SO HUST/CycleGANAging/test', help='The image directory')

@torch.no_grad()
def calculate_metrics(image_true, image_gen):
    """Calculate MSE, SSIM, and PSNR."""
    # Convert tensors to numpy arrays
    image_true = (image_true.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
    image_gen = (image_gen.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0

    # Ensure values are in range [0, 1]
    image_true = np.clip(image_true, 0, 1)
    image_gen = np.clip(image_gen, 0, 1)

    # Calculate MSE
    mse = np.mean((image_true - image_gen) ** 2)

    # Calculate SSIM
    ssim_value = ssim(image_true, image_gen, multichannel=True)

    # Calculate PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * math.log10(1.0 / math.sqrt(mse))

    return mse, ssim_value, psnr

@torch.no_grad()
def main():
    args = parser.parse_args()
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if
                   x.endswith('.png') or x.endswith('.jpg')]
    print(f"Found image paths: {image_paths}")
    if not image_paths:
        print(f"No images found in the directory: {args.image_dir}")
        return
    
    model = Generator(ngf=32, n_residual_blocks=9)
    ckpt = torch.load('pretrained_model/state_dict.pth', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    nr_images = len(image_paths) if len(image_paths) >= 6 else len(image_paths)
    fig, ax = plt.subplots(2, nr_images, figsize=(20, 10))
    random.shuffle(image_paths)

    # Initialize metrics
    total_mse, total_ssim, total_psnr = 0, 0, 0

    for i in range(nr_images):
        img = Image.open(image_paths[i]).convert('RGB')
        img = trans(img).unsqueeze(0)
        aged_face = model(img)

        # Calculate metrics
        mse, ssim_value, psnr = calculate_metrics(img, aged_face)
        total_mse += mse
        total_ssim += ssim_value
        total_psnr += psnr

        # Visualization
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        ax[1, i].imshow(aged_face)

    # Calculate average metrics
    avg_mse = total_mse / nr_images
    avg_ssim = total_ssim / nr_images
    avg_psnr = total_psnr / nr_images

    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f}")

    plt.savefig("mygraph.png")

if __name__ == '__main__':
    main()
