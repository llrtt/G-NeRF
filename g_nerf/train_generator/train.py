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
from training.dataset import Test_Dataset, FFHQ_GEN_Dataset

class LPIPS:
    def __init__(self, device) -> None:
            self.vgg = lpips.LPIPS(net='vgg').to(device)
    
    def loss(self, real, fake):
        loss = self.vgg(real, fake)
        return loss

         

def project(
        device: torch.device,
        training_set,
        G,
        id_encoder,
        num_steps=1000,
        verbose=False,
):
    torch_resize = Resize([224,224])
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore

    for i in G.parameters():
        i.requires_grad = True

    for i in G.superresolution.parameters():
        i.requires_grad = False
    
    lpips = LPIPS(device=device)
    l1_loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam([i for i in G.parameters()], betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)
    for i in range(num_steps):
        for train_data in tqdm(training_set):

            id_images = (torch_resize(train_data["condition_image"]).to(device) / 255.0) * 2 - 1

            id_feature, _ = id_encoder(id_images)
            phase_real_img = (train_data["loss_image"].to(device).to(torch.float32) / 127.5 - 1)
            real_img_raw = filtered_resizing(phase_real_img, size=128, f=upfirdn2d.setup_filter([1,3,3,1], device=device))
            phase_real_c = train_data["loss_c"].to(device)
            ws = G.mapping(z=id_feature[0], c=torch.zeros_like(phase_real_c))
            result = G.synthesis(ws, c=phase_real_c, force_fp32=True)
            synth_images = result['image']
            # loss =l1_loss(synth_images, phase_real_img)
            loss = lpips.loss(synth_images, phase_real_img).mean()
            print(loss)
            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # with torch.no_grad():
            img = (result['image'].permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach()
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'./samples/pti_result/seed{100:04d}.png')

    return G.eval().requires_grad_(False).cpu()

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--encoder_path', 'encoder_path', help='Network pickle filename', required=True)
@click.option('--dataset_path', 'dataset_path', help='Network pickle filename', required=True)
@click.option('--batch_size', 'batch_size', help='Network pickle filename', required=True)
def main(network_pkl: str, 
         encoder_path: str,
         dataset_path: str,
         batch_size: int):
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    G.neural_rendering_resolution = 128
    training_set = FFHQ_GEN_Dataset(path=dataset_path)
    test_set = Test_Dataset(path=dataset_path)
    training_set_iterator = torch.utils.data.DataLoader(dataset=training_set, batch_size=2, pin_memory=True, num_workers=3)
    id_eder = load_pretrained(class_name='training.networks_stylegan2.ResNeXt50',\
                                 path=encoder_path, device=device, pretrained=True)

    G_ema = project(G=G, num_steps=401, device=device, training_set = training_set_iterator, id_encoder=id_eder)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f) # type: ignore
        G['G_ema'] = G_ema
    with open("ffhq_pti.pkl", 'wb') as f:
        pickle.dump(G, f)

if __name__ == "__main__":
    main()
