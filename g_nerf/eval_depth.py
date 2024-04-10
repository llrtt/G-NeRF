

import os
import sys
sys.path.append('/data_local/Deep3DFaceRecon_pytorch/')
import cv2
import json
import glob
import click
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
@click.option('--network', 'network_pkl', help='Network pickle filename', default='/data/baowenjie/20231115_backup/experiments/FFHQ/network-G_ema-3971.pkl')
@click.option('--gpu', type=int, help='gpu', default=0)
@click.option('--id_encoder', help='id_encoder', default='/data/baowenjie/20231115_backup/experiments/FFHQ/network-E-3971.pkl')

def generate_talking_videos(
    network_pkl: str,
    gpu: int, 
    id_encoder: str,
    **kwargs
):
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)

    # torch_resize = Resize([224,224])
    torch_resize_64 = Resize([128,128])
    # get id feature and pose feature
        
    fov = 2 * np.arctan(112. / 1015.) * 180 / np.pi
    renderer = MeshRenderer(rasterize_fov=fov, znear=5., zfar=15., rasterize_size=int(2 * 112.), use_opengl=False)
    bfm_model = bfm.ParametricFaceModel(default_name='/data/baowenjie/20231115_backup/TalkingNeRF/data_preprocessing/Deep3DFaceRecon_pytorch/BFM/BFM_model_front.mat')
    bfm_model.to(device)

    all_mats = glob.glob("/data_local/dataset/FFHQ_in_the_wlid/mesh/*.mat")
    all_mats.sort()

    # load encoders and generator
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
    with dnnlib.util.open_url(id_encoder) as f:
        id_eder = legacy.load_network_pkl(f)['E'].to(device).eval() # type: ignore
    with open('/data_local/dataset/FFHQ_in_the_wlid/label/labels.json', 'r') as f:
        labels = json.load(f)
    all_keys = list(labels.keys())
    avg_ssim = 0
    sum_ssim = 0
    l2_loss = torch.nn.MSELoss()
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    eval_imgs = glob.glob(os.path.join("/data_local/dataset/FFHQ_in_the_wlid/pti_eval", "*.jpg"))

    # for i in tqdm(range(0, 8000)):
    #     mat_path = all_mats[i]
    eval_imgs = glob.glob(os.path.join("/data_local/dataset/FFHQ_in_the_wlid/pti_eval", "*.jpg"))
    for i,path in enumerate(tqdm(eval_imgs)):
        img_basename = os.path.basename(path)
        mat_path = os.path.join("/data_local/dataset/FFHQ_in_the_wlid/mesh", img_basename.replace('.jpg', '.mat'))
        image_path = mat_path.replace('mesh', 'cropped_image').replace('.mat', '.jpg')
        coeff = loadmat(mat_path)
        for key in coeff:
            if key.startswith('__'):
                continue
            else:
                coeff[key] = torch.from_numpy(coeff[key]).to(device)
        
        label_index = os.path.basename(mat_path).replace('.mat', '.png')
        random_index = np.random.randint(0, len(all_keys))
        cam2world_pose_d = LookAtPoseSampler.sample(math.pi/2 - 0.5, math.pi/2, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        label = torch.cat([cam2world_pose_d.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        # label = torch.tensor(labels[all_keys[random_index]],device=device).type(dtype=torch.float32).unsqueeze(0)
        label_for_bfm = label[:, :16].clone().reshape(1, 4, 4)

        pred_vertex, _, pred_color, _ = bfm_model.compute_for_render_test(0, 0, coeff, label_for_bfm)
        pred_mask, depth, _ = renderer(pred_vertex, bfm_model.face_buf, feat=pred_color)
        depth = (depth - (depth)[pred_mask > 0].min()).clamp(0,255)
        depth[pred_mask > 0] = (1 - depth[pred_mask > 0] / depth[pred_mask > 0].max()) * 255
        depth = np.clip(depth.cpu().detach().squeeze(0).permute(1,2,0).numpy(),0,255).astype(np.uint8)
        mask = (pred_mask.squeeze(0).permute(1,2,0).cpu().detach().numpy().astype(np.uint8) * 255)
        depth = cv2.resize(depth, (64,64)).astype(np.float32)
        mask = cv2.resize(mask, (64,64))

        #normalize
        depth[mask > 0] = (depth[mask > 0] - depth[mask > 0].mean()) / depth.std()

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        image = ((torch.from_numpy(image)).to(device) / 255.0).unsqueeze(0) * 2 - 1
        
        id_feature = id_eder(image)
        ws = G.mapping(z=id_feature, c=torch.zeros_like(label))
        output = G.synthesis(ws=ws, c=label, noise_mode='const')
        
        tmp_mask = np.expand_dims(np.expand_dims(mask, 0), 0)
        fake_depth = output['image_depth']
        tmp_mask = torch.from_numpy(tmp_mask)
        fake_depth = (fake_depth - (fake_depth)[tmp_mask > 0].min()).clamp(0,255)
        fake_depth[tmp_mask > 0] = (1 - fake_depth[tmp_mask > 0] / fake_depth[tmp_mask > 0].max()) * 255
        # tmp_depth = ((output['image_depth'] - output['image_depth'].min()) / (output['image_depth'].max() - output['image_depth'].min()) * 255)
        cv2.imwrite("tmp.jpg", (fake_depth.squeeze(0).permute(1,2,0).cpu().detach().numpy()).astype(np.uint8))
        fake_depth[tmp_mask > 0] = (fake_depth[tmp_mask > 0] - fake_depth[tmp_mask > 0].mean()) / fake_depth[tmp_mask > 0].std()
        ssim_val = l2_loss(fake_depth[tmp_mask > 0], torch.from_numpy(np.expand_dims(np.expand_dims(depth, 0), 0))[tmp_mask > 0].to(fake_depth.device))
        # tmp_depth = ((output['image_depth'] - output['image_depth'].min()) / (output['image_depth'].max() - output['image_depth'].min()) * 255)
        sum_ssim += ssim_val
        avg_ssim = avg_ssim * i / (i+1) + ssim_val / (i+1)
        print(avg_ssim)
        print(sum_ssim / (i+1))
        # img = (output['image_raw'][0] * 127.5 + 128).permute(1,2,0).cpu().clamp(0, 255).numpy().astype(np.uint8)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # s_image = (s_image * 127.5 + 128).squeeze().permute(1,2,0).cpu().clamp(0, 255).numpy().astype(np.uint8)
        # s_image = cv2.cvtColor(s_image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('./fake.jpg', img)
        # cv2.imwrite('./real.jpg', s_image)
        # exit()
    print(sum_ssim / 8000)
    

if __name__ == "__main__":
    generate_talking_videos()

