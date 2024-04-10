import cv2
import sys
sys.path.append('talkingnerf')
import os
import glob
import json
import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
# from torchvision.transforms import Resize
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
import legacy

from camera_utils import LookAtPoseSampler
from training.training_loop import load_pretrained
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#----------------------------------------------------------------------------

def evalute_ssim(img_1, img_2) -> float:
    ssim_val = ssim(img_1 * 0.5 + 0.5, img_2 * 0.5 + 0.5, data_range=1.0, size_average=False) # return (N,)
    return ssim_val.mean().item()

def PSNR(img1, img2):
    # img1 and img2 have range [0, 1]
    mse = np.mean((img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

class LPIPS:
    def __init__(self, device) -> None:
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            self.vgg16 = torch.jit.load(f, map_location='cpu').eval().to(device)

    def cal(self, img1, img2):
        target_features = self.vgg16((img1 + 1) * 255 * 0.5, resize_images=True, return_lpips=True)
        synth_features = self.vgg16((img2 + 1) * 255 * 0.5, resize_images=True, return_lpips=True)
        p_loss = (target_features - synth_features).square().sum(1)
        return p_loss.mean().item()


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='experiments/20230430_pzl009_512_resolution_5120_small_pose_re/00006-ffhq-ffhq-gpus2-batch20-gamma5/network-G_ema-latest.pkl')
@click.option('--gpu', type=int, help='gpu', default=0)
@click.option('--id_encoder', help='id_encoder', default='experiments/20230430_pzl009_512_resolution_5120_small_pose_re/00006-ffhq-ffhq-gpus2-batch20-gamma5/network-E-latest.pkl')
@click.option('--resolution', type=int, default=512)

def generate_talking_videos(
    network_pkl: str,
    gpu: int, 
    id_encoder: str,
    resolution: int,
    **kwargs
):
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)

    lpips = LPIPS(device)
    # get id feature and pose feature

    all_paths = glob.glob("/home/huangzixiong/dataset/classnerf/FFHQ_in_the_wlid/cropped_image/*.jpg")
    all_paths.sort()
    # all_paths.reverse()

    # load encoders and generator
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
    with dnnlib.util.open_url(id_encoder) as f:
        id_eder = legacy.load_network_pkl(f)['E'].to(device).eval() # type: ignore
    with open('/home/huangzixiong/dataset/classnerf/FFHQ_in_the_wlid/label/labels.json', 'r') as f:
        labels = json.load(f)
    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * 2)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * 2)

    avg_ssim_raw = 0
    avg_ssim_512 = 0

    avg_psnr_raw = 0
    avg_psnr_512 = 0

    avg_lpips_raw = 0
    avg_lpips_512 = 0

    filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
    for i in tqdm(range(0, 8000)):
        label_index = os.path.basename(all_paths[len(all_paths) - 1 - i]).replace('.jpg', '.png')
        label = torch.tensor(labels[label_index],device=device).type(dtype=torch.float32).unsqueeze(0)

        image = cv2.imread(all_paths[len(all_paths) - 1 - i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        s_image = filtered_resizing((torch.from_numpy(image).to(device).to(torch.float32).unsqueeze(0) / 127.5 - 1), size=128, f=filter)
        image = ((torch.from_numpy(image)).to(device) / 255.0).unsqueeze(0) * 2 - 1
        
        id_feature = id_eder(image)

        ws = G.mapping(z=id_feature, c=torch.zeros_like(label))
        output = G.synthesis(ws=ws, c=label, noise_mode='none')
        
        ssim_raw = evalute_ssim(output['image_raw'], s_image)
        ssim_512 = evalute_ssim(output['image'], image)
        avg_ssim_raw = avg_ssim_raw * i / (i+1) + ssim_raw / (i+1)
        avg_ssim_512 = avg_ssim_512 * i / (i+1) + ssim_512 / (i+1)

        psnr_raw = PSNR(output['image_raw'].squeeze().permute(1,2,0).cpu().numpy(), s_image.squeeze().permute(1,2,0).cpu().numpy())
        psnr_512 = PSNR(output['image'].squeeze().permute(1,2,0).cpu().numpy(), image.squeeze().permute(1,2,0).cpu().numpy())
        avg_psnr_raw = avg_psnr_raw * i / (i+1) + psnr_raw / (i+1)
        avg_psnr_512 = avg_psnr_512 * i / (i+1) + psnr_512 / (i+1)

        lpips_raw = lpips.cal(output['image_raw'], s_image)
        lpips_512 = lpips.cal(output['image'], image)
        avg_lpips_raw = avg_lpips_raw * i / (i+1) + lpips_raw / (i+1)
        avg_lpips_512 = avg_lpips_512 * i / (i+1) + lpips_512 / (i+1)

        print(f'avg_ssim_raw: {avg_ssim_raw}, avg_ssim_512: {avg_ssim_512}, avg_psnr_raw: {avg_psnr_raw}, avg_psnr_512: {avg_psnr_512}, avg_lpips_raw: {avg_lpips_raw}, avg_lpips_512: {avg_lpips_512}')

        label_index = label_index.replace('.png', '')
        if resolution == 128:
            img = (output['image_raw'][0] * 127.5 + 128).permute(1,2,0).cpu().clamp(0, 255).numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            s_image = (s_image.squeeze(0) * 127.5 + 128).squeeze().permute(1,2,0).cpu().clamp(0, 255).numpy().astype(np.uint8)
            s_image = cv2.cvtColor(s_image, cv2.COLOR_RGB2BGR)
            label_index = label_index.replace('.png', '')
            cv2.imwrite('./evaluation_results/fake/128_pre/'+ label_index + '.jpg', img)
            # cv2.imwrite('./evaluation_results/real/128/'+ label_index + '.jpg', s_image)
        else:
            img = (output['image'][0] * 127.5 + 128).permute(1,2,0).cpu().clamp(0, 255).numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image = (image.squeeze(0) * 127.5 + 128).squeeze().permute(1,2,0).cpu().clamp(0, 255).numpy().astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite('./evaluation_results/fake/512_pre/'+ label_index + '.jpg', img)
            # cv2.imwrite('./evaluation_results/real/512/'+ label_index + '.jpg', image)
    

if __name__ == "__main__":
    generate_talking_videos()

