import cv2
import os
import glob
import json
import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
import math
from tqdm import tqdm
from torchvision.transforms import Resize
import legacy
import random
import skimage
from insightface.app import FaceAnalysis 

from camera_utils import LookAtPoseSampler
from gen_talking_videos import normalize_depth
from training.training_loop import load_pretrained
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#----------------------------------------------------------------------------
def evalute_ssim(img_1, img_2) -> float:
    ssim_val = ssim(img_1, img_2, data_range=2.0, size_average=False) # return (N,)
    return ssim_val.mean().item()

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
@click.option('--network', 'network_pkl', help='Network pickle filename', default='experiments/202300604_pzl009_cars_128_5120_final/00000-shapenet-shapenet-gpus2-batch24-gamma5/network-G_ema-best.pkl')
@click.option('--dataset', 'dataset', type=click.Choice(['ffhq', 'afhqv2', 'celeba']), help='whether to use image prepared in a folder', default='ffhq')
@click.option('--id_encoder', help='id_encoder', default='experiments/202300604_pzl009_cars_128_5120_final/00000-shapenet-shapenet-gpus2-batch24-gamma5/network-E-best.pkl')
@click.option('--outdir', 'outdir', help='Identity reference', default='evaluation_results/quanitative_results/srn_chairs')
@click.option('--dataset_dir', help='id_encoder', default='/home/huangzixiong/dataset/classnerf/srn/cars_test')
@click.option('--res', help='Output directory', type=int, default=64, metavar='DIR')
@click.option('--gpu', type=int, help='gpu', default=0)
@click.option('--label_path', help='label_path', default='/home/huangzixiong/dataset/classnerf/srn/cars_test/label/labels.json')
@click.option('--isRand', 'isRand', help='isRand', default=True)
@click.option('--only_depth', 'only_depth', help='isRand', default=False)
def generate_talking_videos(
    network_pkl: str,
    dataset: bool,
    outdir: str,
    dataset_dir: str, 
    id_encoder: str,
    res: int,
    gpu: int,
    label_path: str,
    isRand: bool,
    only_depth: bool,
):
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)
    lpips = LPIPS(device)
    # torch_resize = Resize([224,224])
    torch_resize_64 = Resize([res,res], antialias=True)
    
    with open(label_path, 'r') as f:
        labels = json.load(f)

    # load encoders and generator
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
    with dnnlib.util.open_url(id_encoder) as f:
        id_eder = legacy.load_network_pkl(f)['E'].to(device).eval() # type: ignore

    avg_ssim_raw = 0
    avg_ssim_512 = 0

    avg_psnr_raw = 0
    avg_psnr_512 = 0

    avg_lpips_raw = 0
    avg_lpips_512 = 0

    counter = 0
    all_categories = os.listdir(dataset_dir)
    all_categories = [os.path.join(dataset_dir, x) for x in all_categories if os.path.isdir(os.path.join(dataset_dir, x, 'rgb'))]
    for category in tqdm(all_categories):
        all_img_path = glob.glob(os.path.join(category, 'rgb', '*.png'))
        all_img_path.sort()
        input_img = cv2.imread(all_img_path[0])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        input_img = ((torch.from_numpy(input_img)).to(device) / 255.0).unsqueeze(0) * 2 - 1
        gt_index = np.random.randint(0, len(all_img_path), 10)
        # gt_index = [0]
        category_name = category.split('/')[-1]
        outdir_fake_raw = os.path.join(outdir, category_name, 'fake', str(res))
        outdir_real_raw = os.path.join(outdir, category_name, 'real', str(res))
        outdir_real_512 = os.path.join(outdir, category_name, 'real', '128')
        outdir_fake_512 = os.path.join(outdir, category_name, 'fake', '128')
        for dir in [outdir_fake_raw, outdir_real_raw, outdir_real_512, outdir_fake_512]:
            if not os.path.exists(dir):
                os.makedirs(dir)
        for i in gt_index:
            gt_img = cv2.imread(all_img_path[i])
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            gt_img = ((torch.from_numpy(gt_img)).to(device) / 255.0).unsqueeze(0) * 2 - 1
            s_image = torch_resize_64(gt_img)
            label_index = (all_img_path[i]).replace(dataset_dir, '')[1:]
            gt_label = torch.tensor(labels[label_index],device=device).type(dtype=torch.float32).unsqueeze(0)
            id_feature = id_eder(input_img)
            output = G(z=id_feature, c=gt_label, noise_mode='const', neural_rendering_resolution=res)

            # ssim_raw = evalute_ssim(output['image_raw'], s_image)
            # ssim_512 = evalute_ssim(output['image'].cpu(), gt_img.cpu())
            ssim_raw = skimage.metrics.structural_similarity(np.clip(output['image_raw'].squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1), np.clip(s_image.squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1),data_range=1, multichannel=True, channel_axis=2)
            ssim_512 = skimage.metrics.structural_similarity(np.clip(output['image'].squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1), np.clip(gt_img.squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1),data_range=1, multichannel=True, channel_axis=2)
            avg_ssim_raw = avg_ssim_raw * counter / (counter+1) + ssim_raw / (counter+1)
            avg_ssim_512 = avg_ssim_512 * counter / (counter+1) + ssim_512 / (counter+1)

            psnr_raw = skimage.metrics.peak_signal_noise_ratio(np.clip(output['image_raw'].squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1), np.clip(s_image.squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1),data_range=1)
            psnr_512 = skimage.metrics.peak_signal_noise_ratio(np.clip(output['image'].squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1), np.clip(gt_img.squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1),data_range=1)
            avg_psnr_raw = avg_psnr_raw * counter / (counter+1) + psnr_raw / (counter+1)
            avg_psnr_512 = avg_psnr_512 * counter / (counter+1) + psnr_512 / (counter+1)

            lpips_raw = lpips.cal(output['image_raw'], s_image)
            lpips_512 = lpips.cal(output['image'], gt_img)
            avg_lpips_raw = avg_lpips_raw * counter / (counter+1) + lpips_raw / (counter+1)
            avg_lpips_512 = avg_lpips_512 * counter / (counter+1) + lpips_512 / (counter+1)
            counter+=1
            print(f'avg_ssim_raw: {avg_ssim_raw}, avg_ssim_512: {avg_ssim_512}, avg_psnr_raw: {avg_psnr_raw}, avg_psnr_512: {avg_psnr_512}, avg_lpips_raw: {avg_lpips_raw}, avg_lpips_512: {avg_lpips_512}')
            

if __name__ == "__main__":
    generate_talking_videos()
