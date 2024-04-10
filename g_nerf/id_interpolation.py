import cv2
import json
import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
from torchvision.transforms import Resize

import legacy

from camera_utils import LookAtPoseSampler
from training.training_loop import load_pretrained
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='experiments/ablation_studies/20230510_pzl009_512_5120_final/00003-ffhq-ffhq-gpus2-batch24-gamma5/network-G_ema-best.pkl')
@click.option('--gpu', type=int, help='gpu', default=0)
@click.option('--id_encoder', help='id_encoder', default='experiments/ablation_studies/20230510_pzl009_512_5120_final/00003-ffhq-ffhq-gpus2-batch24-gamma5/network-E-best.pkl')
@click.option('--id_1', help='id_1', default='/home/huangzixiong/dataset/classnerf/FFHQ_in_the_wlid/cropped_image/00003.jpg')
@click.option('--id_2', help='id_2', default='/home/huangzixiong/dataset/classnerf/FFHQ_in_the_wlid/cropped_image/00018.jpg')
@click.option('--id_3', help='id_2', default='/home/huangzixiong/dataset/classnerf/FFHQ_in_the_wlid/cropped_image/28710.jpg')
@click.option('--fine_id', help='fine_id', default='/home/huangzixiong/dataset/classnerf/FFHQ_in_the_wlid/cropped_image/64487.jpg')
def generate_talking_videos(
    network_pkl: str,
    gpu: int, 
    id_encoder: str,
    fine_id: str,
    id_1: str,
    id_2: str,
    id_3: str,
    **kwargs
):
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)

    id_images = []
    for id in [id_1, id_2, id_3, fine_id]:
        id_image = cv2.imread(id)
        id_image = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        id_image = ((torch.from_numpy(id_image)).to(device) / 255.0).unsqueeze(0) * 2 - 1
        id_images.append(id_image)

    # load encoders and generator
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
    with dnnlib.util.open_url(id_encoder) as f:
        id_eder = legacy.load_network_pkl(f)['E'].to(device).eval() # type: ignore
    # with open('/mnt/cephfs/dataset/Face/FFHQ_in_the_wlid/label/labels.json', 'r') as f:
    #     labels = json.load(f)

    # get id features
    fine_id_feature = id_eder(id_images[-1])
    
    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    cam2world_pose_s = LookAtPoseSampler.sample(3.14/2, 3.14/2,\
                                                camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    c_s = torch.cat([cam2world_pose_s.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c_s = c_s.repeat(fine_id_feature.shape[0], 1)

    fine_id_ws = G.mapping(z=fine_id_feature, c=torch.zeros_like(c_s))

    all_imgs = []
    for i in range(14):
        imgs = []
        for id_image in id_images[:-1]:
            id_feature = id_eder(id_image)
            ws = G.mapping(z=id_feature, c=torch.zeros_like(c_s))
            w = torch.zeros_like(ws)
            w[:, :i] = ws[:, :i]
            w[:, i:] = fine_id_ws[:, i:]
            # fine_id_ws[:, :i] = ws[:, :i]
            # fine_id_ws[:, i:] = fine_id_ws[:, i:]
            output = G.synthesis(ws=w, c=c_s, noise_mode='const')
            img = (output['image'][0] * 127.5 + 128).permute(1,2,0).cpu().clamp(0, 255).numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            imgs.append(img)
        imgs = np.concatenate(imgs, 1)
        all_imgs.append(imgs)
    all_imgs = np.concatenate(all_imgs, 0)
    cv2.imwrite('./interpolation.jpg', all_imgs)
        # exit()
    

if __name__ == "__main__":
    generate_talking_videos()

