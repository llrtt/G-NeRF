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
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.0005,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        verbose=False,
        device: torch.device,
        use_wandb=True,
        initial_w=None,
        image_log_step=global_config.image_rec_result_log_snapshot,
        w_name: str,
        camera_params: torch.tensor,
        w_path: torch.tensor
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
    w_opt = w_path
    w_opt.requires_grad = False
    ws = w_opt
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)
    l1_loss = torch.nn.L1Loss()
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
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
        p_loss = loss_fn_vgg(synth_images, target_images)
        loss = l1_loss(synth_images, target_images) + p_loss
        # target_features = vgg16((target_images + 1) * 255 * 0.5, resize_images=True, return_lpips=True)
        # synth_features = vgg16((result['image'] + 1) * 255 * 0.5, resize_images=True, return_lpips=True)
        # loss = (target_features - synth_features).square().sum()
        # synth_images = result['image_raw']
        # p_loss_raw = loss_fn_vgg(synth_images, real_img_raw)
        # loss =  p_loss + p_loss_raw
        
        # if step % image_log_step == 0:
        #     
        #         if use_wandb:
        #             global_config.training_step += 1
                    # log_utils.log_image_from_w((w_opt), G, w_name, camera_params)
        with torch.no_grad():
            img = G.synthesis(ws, camera_params, force_fp32=True)['image']
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach()
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'./samples/pti_result/seed{100:04d}.png')

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return G.eval().requires_grad_(False).cpu()

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--img_path', 'img_path', help='Network pickle filename', required=True)
@click.option('--encoder_path', 'encoder_path', help='Network pickle filename', required=True)
def main(network_pkl: str, img_path: str, encoder_path: str):
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    with open('/mnt/cephfs/dataset/Face/FFHQ_in_the_wlid/label/labels.json', 'r') as f:
        labels = json.load(f)
    label_index = os.path.basename(img_path).replace('.jpg', '.png')
    label = torch.tensor(labels[label_index],device=device).type(dtype=torch.float32).unsqueeze(0)
    torch_resize = Resize([224,224])
    id_eder = load_pretrained(class_name='training.networks_stylegan2.ResNeXt50',\
                                 path=encoder_path, device=device)
    image = cv2.imread(os.path.join(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    id_image = (torch_resize(torch.from_numpy(image)).to(device) / 255.0).unsqueeze(0) * 2 - 1
    id_feature, _ = id_eder(id_image)
    w = G.mapping(id_feature[0], c=torch.zeros_like(label))

    image = np.array(PIL.Image.open(os.path.join(img_path)))
    image = torch.from_numpy(image.transpose(2, 0, 1)).to(device).to(torch.float32) / 127.5 - 1 # HWC => CHW

    G_ema = project(G=G, target=image, num_steps=401, camera_params=label, device=device, w_name='20230226', w_path=w)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f) # type: ignore
        G['G_ema'] = G_ema
    with open("ffhq_pti.pkl", 'wb') as f:
        pickle.dump(G, f)

if __name__ == "__main__":
    main()
