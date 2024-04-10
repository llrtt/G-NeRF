# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import os
import cv2
import time
import sys
import glob
sys.path.append('talkingnerf')
import PIL
import pickle
import json
import copy
import click
import legacy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from configs import global_config, hyperparameters
from utils import log_utils
import dnnlib
import lpips
import math
import multiprocess as mp
from camera_utils import LookAtPoseSampler
from training.training_loop import load_pretrained
from torchvision.transforms import Resize
from training.dual_discriminator import filtered_resizing
from torch_utils.ops import upfirdn2d

device = torch.device('cuda')
url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
with dnnlib.util.open_url(url) as f:
    vgg16 = torch.jit.load(f).eval().to(device)
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

def project(
        G,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=350,
        initial_learning_rate=0.0005,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        verbose=False,
        device: torch.device,
        camera_params: torch.tensor,
        train_w: torch.tensor
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)
    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore

    for i in G.parameters():
        i.requires_grad = True
    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    real_img_raw = filtered_resizing(target_images, size=128, f=upfirdn2d.setup_filter([1,3,3,1], device=device))
    ws = train_w

    target_features = vgg16((target_images+1)*127.5, resize_images=True, return_lpips=True)
    # Setup noise inputs.
    optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)
    for step in tqdm(range(num_steps)):
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        result = G.synthesis(ws, camera_params, force_fp32=True)
        synth_images = result['image']
        synth_images = (synth_images + 1) * (255 / 2)
        # if synth_images.shape[2] > 256:
        #     synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=True, return_lpips=True)
        dist = (target_features - synth_features).square().sum()
        loss = dist
        # print(loss.item())
        # with torch.no_grad():
        #     img = G.synthesis(ws, camera_params, force_fp32=True)['image']
        #     img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach()
        #     PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'./samples/pti_result/seed{100:04d}.png')

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if loss <= 0.33:
            break

    return G.eval().requires_grad_(False)

def train_w(G, image, label, opt_w, num_steps, device):
    l1_loss = torch.nn.L1Loss()
    lr_rampdown_length = 0.25
    lr_rampup_length = 0.05
    initial_learning_rate = 0.005
    opt_w = opt_w.unsqueeze(1).repeat([1, G.backbone.mapping.num_ws, 1]).requires_grad_(True)
    optimizer = torch.optim.Adam([opt_w], betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)

    for step in tqdm(range(num_steps)):
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        result = G.synthesis(opt_w, label, force_fp32=True)
        synth_images = result['image']
        p_loss = loss_fn_vgg(synth_images, image)
        loss = l1_loss(synth_images, image) + p_loss
        # print(loss)
        # with torch.no_grad():
        #     img = G.synthesis(opt_w, label, force_fp32=True)['image']
        #     img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach()
        #     PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'./samples/pti_result/seed{100:04d}.png')

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return opt_w


# @click.command()
# @click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# @click.option('--img_path', 'img_path', help='Network pickle filename', required=True)
# @click.option('--w_path', 'w_path', help='Network pickle filename', required=True)
# @click.option('--skip_w', 'skip_w', help='Network pickle filename', default=None)
# @click.option('--out_path', 'out_path', help='Network pickle filename', default=None)
def save_rgb_image(img, base_name, res):
    rec_path = '/data/baowenjie/20231115_backup/evaluation_results/pti_8000_032/resonstruction'
    random_path = '/data/baowenjie/20231115_backup/evaluation_results/pti_8000_032/random'
    side_path = '/data/baowenjie/20231115_backup/evaluation_results/pti_8000_032/side_face'
    img_512 = img.detach().cpu().numpy()
    img_512 = np.clip(img_512 * 127.5 + 128, 0, 255).astype(np.uint8)
    img_512 = img_512.transpose(0, 2, 3, 1)
    img_512 = img_512[:, :, :, ::-1]
    for i, path in enumerate([rec_path, random_path, side_path]):
        os.makedirs(os.path.join(path, str(res)), exist_ok=True)
        cv2.imwrite(os.path.join(path, str(res), base_name), img_512[i])

def normalize_depth(depth, range):
    hi, lo = range
    depth = (depth - lo) * (255 / (hi - lo))
    return depth.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

def save_depth_image(img, base_name, res):
    rec_path = '/data/baowenjie/20231115_backup/evaluation_results/pti_8000_032/resonstruction'
    random_path = '/data/baowenjie/20231115_backup/evaluation_results/pti_8000_032/random'
    side_path = '/data/baowenjie/20231115_backup/evaluation_results/pti_8000_032/side_face'
    img_depth = img.detach().cpu().numpy()
    depth_map = [normalize_depth(i, [i.max(), i.min()]) for i in -img]
    for i, path in enumerate([rec_path, random_path, side_path]):
        os.makedirs(os.path.join(path, 'depth'), exist_ok=True)
        np.save(os.path.join(path, 'depth', base_name.replace('.jpg', '.npy')), img_depth[i])
        cv2.imwrite(os.path.join(path, 'depth', base_name), depth_map[i])

def main(network_pkl, img_path, w_path, skip_w, out_path, labels, G):
    device = torch.device('cuda')
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    label_index = os.path.basename(img_path).replace('.jpg', '.png')
    all_keys = list(labels.keys())
    label = torch.tensor(labels[label_index],device=device).type(dtype=torch.float32).unsqueeze(0)

    w = G.backbone.mapping.w_avg.unsqueeze(0)
    image = np.array(PIL.Image.open(os.path.join(img_path)))
    image = torch.from_numpy(image.transpose(2, 0, 1)).to(device).to(torch.float32) / 127.5 - 1 # HWC => CHW
    if os.path.isfile(w_path):
        trained_w = torch.from_numpy(np.load(w_path)).to(device)
    else:
        trained_w = train_w(G, image, label, w, device=device, num_steps=300)
        np.save(w_path, trained_w.cpu().detach().numpy())

    G_ema = project(G=G, target=image, camera_params=label, device=device, train_w=trained_w)
    # print(f'Elapsed: {(time.time() - start):.1f} s')
    # with dnnlib.util.open_url(network_pkl) as f:
    #     G = legacy.load_network_pkl(f) # type: ignore
    #     G['G_ema'] = G_ema
    # with open(out_path, 'wb') as f:
    #     pickle.dump(G, f)

    # reconstruction
    label_rec = label
    # random
    random_index = all_keys[np.random.randint(0, len(all_keys))]
    label_random = torch.tensor(labels[random_index],device=device).type(dtype=torch.float32).unsqueeze(0)
    # side face
    cam2world_pose_d = LookAtPoseSampler.sample(math.pi/2 - 0.5, math.pi/2, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    label_side = torch.cat([cam2world_pose_d.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    # concat labels
    label = torch.cat([label_rec, label_random, label_side], 0)
    ws = trained_w.repeat([3, 1, 1])
    # generate images
    result = G_ema.synthesis(ws=ws, c=label, noise_mode='const', neural_rendering_resolution=64)
    # save images (reconstruction, random, side)
    base_name = os.path.basename(img_path)
    save_rgb_image(result['image'], base_name, 512)
    save_rgb_image(result['image_raw'], base_name, 64)
    save_depth_image(result['image_depth'], base_name, 64)


def multi_training(image_paths,rank,nproc):
    with dnnlib.util.open_url('ffhqrebalanced512-64.pkl') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    with open('/data_local/dataset/FFHQ_in_the_wlid/label/labels.json', 'r') as f:
        labels = json.load(f)
    for image_path in tqdm(image_paths):
        base_name = os.path.basename(image_path)
        side_depth_path = os.path.join('/data/baowenjie/20231115_backup/evaluation_results/pti_8000_032/side_face/depth', base_name)
        if os.path.isfile(side_depth_path):
            continue
        w_path = os.path.join('/data/baowenjie/20231115_backup/TalkingNeRF/pti_result/pti_8000/latent_code', base_name.replace('.jpg', '.npy'))
        model_out_path = os.path.join('./pti_result/checkpoint', base_name.replace('.jpg', '.pkl'))
        main(network_pkl='ffhqrebalanced512-64.pkl', img_path=image_path, w_path=w_path, skip_w=None, out_path=model_out_path, labels=labels, G=G)

if __name__ == "__main__":
    all_imgs = glob.glob('/data_local/dataset/FFHQ_in_the_wlid/cropped_image/*.jpg')
    all_imgs.sort()
    all_imgs.reverse()
    all_imgs = all_imgs[:4000]
    multi_training(all_imgs,0,1)
    # nproc = 4
    # p = mp.Pool(nproc)
    # for i in range(nproc):
    #     p.apply_async(multi_training, args=(all_imgs[i::nproc],i,nproc))
    # p.close()
    # p.join()