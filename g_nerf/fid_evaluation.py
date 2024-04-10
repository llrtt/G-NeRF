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
from PC_AVS_audio_preprocess import *

from camera_utils import LookAtPoseSampler
from training.training_loop import load_pretrained
#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--gpu', type=int, help='gpu', default=0)
@click.option('--id_encoder', help='id_encoder', default='/mnt/cephfs/home/llrt/TalkingNeRF/experiments/20230218_gpu020_train_encoder/00001-ffhq-eg3d-gpus4-batch16-gamma1/network-snapshot-001800.pkl')

def generate_talking_videos(
    network_pkl: str,
    gpu: int, 
    id_encoder: str,
    **kwargs
):
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)

    torch_resize = Resize([224,224])

    # get id feature and pose feature

    all_paths = glob.glob("/mnt/cephfs/dataset/Face/FFHQ_in_the_wlid/cropped_image/*.jpg")
    all_paths.sort()
    all_paths.reverse()

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * 2)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * 2)
    with open('/mnt/cephfs/dataset/Face/FFHQ_in_the_wlid/label/dataset.json', 'r') as f:
        labels = json.load(f)['labels']

    id_eder = load_pretrained(class_name='training.networks_stylegan2.ResNeXt50',\
                                 path=id_encoder, device=device) 

    for i in tqdm(range(0, 8000)):
        label_index = np.random.randint(0, 8000)
        label = torch.tensor(labels[label_index][1], device=device).type(dtype=torch.float32).unsqueeze(0)
        # all_gen_z = torch.randn([1, G.z_dim], device=device)

        # label_index = np.random.randint(0, 8000)
        image = cv2.imread(all_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        image = (torch_resize(torch.from_numpy(image)).to(device) / 255.0).unsqueeze(0) * 2 - 1
        
        id_feature, _ = id_eder(image)
        style = id_feature[0]

        ws = G.mapping(z=style, c=label)
        output = G.synthesis(ws=ws, c=label, noise_mode='const')

        img = (output['image'][0] * 127.5 + 128).permute(1,2,0).cpu().clamp(0, 255).squeeze().numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('/mnt/cephfs/dataset/Face/FFHQ_in_the_wlid/eval_image/fake/' + str(i).zfill(5) + '.jpg', img)
        # exit()
    

if __name__ == "__main__":
    generate_talking_videos()

