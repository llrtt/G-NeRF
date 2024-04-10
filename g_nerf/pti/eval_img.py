

import os
import sys
sys.path.append('/data_local/Deep3DFaceRecon_pytorch/')
import cv2
import json
import glob
import click
import sys
sys.path.append('talkingnerf')
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
import math
import models.bfm as bfm
from tqdm import tqdm
from scipy.io import loadmat
from torchvision.transforms import Resize
from util.nvdiffrast import MeshRenderer

import legacy

from camera_utils import LookAtPoseSampler
from training.training_loop import load_pretrained
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='/data/baowenjie/20231115_backup/experiments/Ablation/only_multi-view/network-G_ema-best.pkl')
@click.option('--gpu', type=int, help='gpu', default=0)
@click.option('--id_encoder', help='id_encoder', default='/data/baowenjie/20231115_backup/experiments/Ablation/only_multi-view/network-E-best.pkl')

def generate_talking_videos(
    network_pkl: str,
    gpu: int, 
    id_encoder: str,
    **kwargs
):
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)
        
    fov = 2 * np.arctan(112. / 1015.) * 180 / np.pi
    renderer = MeshRenderer(rasterize_fov=fov, znear=5., zfar=15., rasterize_size=int(2 * 112.), use_opengl=False)
    bfm_model = bfm.ParametricFaceModel(default_name='/data/baowenjie/20231115_backup/TalkingNeRF/data_preprocessing/Deep3DFaceRecon_pytorch/BFM/BFM_model_front.mat')
    bfm_model.to(device)

    all_mats = glob.glob("/data_local/dataset/FFHQ_in_the_wlid/mesh/*.mat")
    all_mats.sort()

    with open('/data_local/dataset/FFHQ_in_the_wlid/label/labels.json', 'r') as f:
        labels = json.load(f)
    all_keys = list(labels.keys())
    avg_ssim = 0
    sum_ssim = 0
    l2_loss = torch.nn.MSELoss()
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    eval_imgs = glob.glob(os.path.join("/data_local/dataset/EG3D_GEN_DEPTH_EVAL", "*.jpg"))
    avg_depth_64 = 0
    counter = 0
    for i,path in enumerate(tqdm(eval_imgs)):
        if 'side_face' in path:
        # if True:
            mat_path = os.path.join("/data_local/dataset/EG3D_GEN_DEPTH_EVAL_GT", os.path.basename(path).replace('.jpg', '.mat'))
            coeff = loadmat(mat_path)
            for key in coeff:
                if key.startswith('__'):
                    continue
                else:
                    coeff[key] = torch.from_numpy(coeff[key]).to(device)
            cam2world_pose_d = LookAtPoseSampler.sample(math.pi/2 - 0.5, math.pi/2, radius=2.7, device=device)
            label = torch.cat([cam2world_pose_d.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            label_for_bfm = label[:, :16].clone().reshape(1, 4, 4)
            pred_vertex, _, pred_color, _ = bfm_model.compute_for_render_test(0, 0, coeff, label_for_bfm)
            pred_mask, depth, _ = renderer(pred_vertex, bfm_model.face_buf, feat=pred_color)
            depth = (depth - (depth)[pred_mask > 0].min()).clamp(0,255)
            depth[pred_mask > 0] = (1 - depth[pred_mask > 0] / depth[pred_mask > 0].max()) * 255
            depth = np.clip(depth.cpu().detach().squeeze(0).permute(1,2,0).numpy(),0,255).astype(np.uint8)
            mask = (pred_mask.squeeze(0).permute(1,2,0).cpu().detach().numpy().astype(np.uint8) * 255)
            depth = cv2.resize(depth, (64,64)).astype(np.float32)
            mask = cv2.resize(mask, (64,64))
            depth[mask > 0] = (depth[mask > 0] - depth[mask > 0].mean()) / depth.std()
        else:
            depth_path = os.path.join('/data_local/dataset/EG3D_GEN_DEPTH_EVAL_GT', os.path.basename(eval_imgs[i]).replace('.jpg', '_depth.jpg'))
            if not os.path.isfile(depth_path):
                continue
            mask = cv2.imread(depth_path.replace('depth.', 'mask.'), 0)
            mask = cv2.resize(mask, (64,64))

            depth = cv2.imread(depth_path, 0)
            depth = cv2.resize(depth, (64,64)).astype(np.float32)
            depth[mask > 0] = (depth[mask > 0] - depth[mask > 0].mean()) / depth.std()

        tmp_mask = np.expand_dims(np.expand_dims(mask, 0), 0)
        tmp_mask = torch.from_numpy(tmp_mask)

        fake_depth = np.load(path.replace('.jpg', '.npy')).astype(np.float32)
        fake_depth = torch.from_numpy(fake_depth).to(device).unsqueeze(0)
        fake_depth = (fake_depth - (fake_depth)[tmp_mask > 0].min()).clamp(0,255)
        fake_depth[tmp_mask > 0] = (1 - fake_depth[tmp_mask > 0] / fake_depth[tmp_mask > 0].max()) * 255
        fake_depth[tmp_mask > 0] = (fake_depth[tmp_mask > 0] - fake_depth[tmp_mask > 0].mean()) / fake_depth[tmp_mask > 0].std()
        depth_loss = l2_loss(torch.from_numpy(np.expand_dims(np.expand_dims(depth, 0), 0))[tmp_mask > 0].to(device), \
                             fake_depth[tmp_mask > 0])
        avg_depth_64 = avg_depth_64 * counter / (counter+1) + depth_loss / (counter+1)
        counter += 1
        print(avg_depth_64)
    print(sum_ssim / 100)
    

if __name__ == "__main__":
    generate_talking_videos()

