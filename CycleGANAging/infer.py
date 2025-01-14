import os
import random
from argparse import ArgumentParser
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity
import numpy as np

from gan_module import Generator

parser = ArgumentParser()
parser.add_argument(
    '--image_dir', default='CycleGANAging/Test Dataset - Aligned and Cropped', help='The image directory')
parser.add_argument(
    '--output_dir', default='Output/output_images', help='Directory to save the output images')

@torch.no_grad()
def calculate_metrics(image_true, image_gen):
    image_true_np = image_true.squeeze().permute(1, 2, 0).cpu().numpy()
    image_gen_np = image_gen.squeeze().permute(1, 2, 0).cpu().numpy()
    
    data_range = image_true_np.max() - image_true_np.min()
    win_size = min(image_true_np.shape[0], image_true_np.shape[1], 7)
    
    ssim_value = structural_similarity(
        image_true_np, 
        image_gen_np, 
        multichannel=True, 
        data_range=data_range,
        win_size=win_size,  
        channel_axis=2      
    )
    
    mse = ((image_true_np - image_gen_np) ** 2).mean()
    psnr = 20 * torch.log10(data_range / torch.sqrt(torch.tensor(mse)))
    
    return mse, ssim_value, psnr

@torch.no_grad()
def main():
    args = parser.parse_args()
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if
                   x.endswith('.png') or x.endswith('.jpg')]
    if not image_paths:
        print(f"No images found in the directory: {args.image_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    model = Generator(ngf=32, n_residual_blocks=9)
    ckpt = torch.load('C:/Users/phamd/OneDrive/Documents/GitHub/Face-Aging/CycleGANAging/pretrained_model/state_dict.pth', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    total_mse, total_ssim, total_psnr = 0, 0, 0
    for i, path in enumerate(image_paths):
        img = Image.open(path).convert('RGB')
        img_tensor = trans(img).unsqueeze(0)
        aged_face = model(img_tensor)

        mse, ssim_value, psnr = calculate_metrics(img_tensor, aged_face)
        total_mse += mse
        total_ssim += ssim_value
        total_psnr += psnr

        # Convert tensors back to image format
        original_image = ((img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        aged_image = ((aged_face.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        
        # Save each output
        original_output_path = os.path.join(args.output_dir, f"original_{i}.png")
        aged_output_path = os.path.join(args.output_dir, f"aged_{i}.png")

        Image.fromarray(original_image).save(original_output_path)
        Image.fromarray(aged_image).save(aged_output_path)

    avg_mse = total_mse / len(image_paths)
    avg_ssim = total_ssim / len(image_paths)
    avg_psnr = total_psnr / len(image_paths)
    
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f}")

if __name__ == '__main__':
    main()
