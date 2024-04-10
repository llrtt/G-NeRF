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
from camera_utils import LookAtPoseSampler
from training.training_loop import load_pretrained
from torchvision.transforms import Resize
from training.dual_discriminator import filtered_resizing
from torch_utils.ops import upfirdn2d
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
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

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
        print(loss.item())
        with torch.no_grad():
            img = G.synthesis(ws, camera_params, force_fp32=True)['image']
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach()
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'./samples/pti_result/seed{100:04d}.png')

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # if loss < 0.28728:
        #     break

    return G.eval().requires_grad_(False).cpu()

def train_w(G, image, label, opt_w, num_steps, device):
    l1_loss = torch.nn.L1Loss()
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
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
        print(loss)
        with torch.no_grad():
            img = G.synthesis(opt_w, label, force_fp32=True)['image']
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach()
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'./samples/pti_result/seed{100:04d}.png')

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return opt_w


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--img_path', 'img_path', help='Network pickle filename', required=True)
@click.option('--w_path', 'w_path', help='Network pickle filename', required=True)
@click.option('--skip_w', 'skip_w', help='Network pickle filename', default=None)

def main(network_pkl: str, img_path: str, w_path: str, skip_w: str):
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    with open('/data_local/dataset/FFHQ_in_the_wlid/label/labels.json', 'r') as f:
        labels = json.load(f)
    label_index = os.path.basename(img_path).replace('.jpg', '.png')
    label = torch.tensor(labels[label_index],device=device).type(dtype=torch.float32).unsqueeze(0)

    w = G.backbone.mapping.w_avg.unsqueeze(0)
    image = np.array(PIL.Image.open(os.path.join(img_path)))
    image = torch.from_numpy(image.transpose(2, 0, 1)).to(device).to(torch.float32) / 127.5 - 1 # HWC => CHW
    start = time.time()
    if skip_w is not None:
        trained_w = torch.from_numpy(np.load(skip_w)).to(device)
    else:
        trained_w = train_w(G, image, label, w, device=device, num_steps=600)
    np.save(w_path, trained_w.cpu().detach().numpy())

    G_ema = project(G=G, target=image, camera_params=label, device=device, train_w=trained_w)
    print(f'Elapsed: {(time.time() - start):.1f} s')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f) # type: ignore
        G['G_ema'] = G_ema
    with open("ffhq_pti.pkl", 'wb') as f:
        pickle.dump(G, f)

if __name__ == "__main__":
    main()
