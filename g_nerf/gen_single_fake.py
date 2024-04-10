import os
import cv2
import json
import click
import dnnlib
import imageio
import numpy as np
import torch
from tqdm import tqdm
from gen_talking_videos import normalize_depth
import legacy

from camera_utils import LookAtPoseSampler
#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='experiments/20230506_pzl009_cats_64_512_5120_refine_super_refine_pose/00004-afhqv2-afhqv2-gpus2-batch24-gamma5/network-G_ema-latest.pkl')
@click.option('--id_image', 'id_image', help='Identity reference', required=True)
@click.option('--id_encoder', help='id_encoder', default='experiments/20230506_pzl009_cats_64_512_5120_refine_super_refine_pose/00004-afhqv2-afhqv2-gpus2-batch24-gamma5/network-E-latest.pkl')
@click.option('--gpu', type=int, help='gpu', default=0)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')

def generate_talking_videos(
    network_pkl: str,
    id_image: str,
    id_encoder: str,
    gpu: int, 
    outdir: str
):
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)
    # preprocess id image
    print('Loading image from "%s"...' % id_image)
    image = cv2.imread(id_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    image = ((torch.from_numpy(image)).to(device) / 255.0).unsqueeze(0) * 2 - 1

    # load encoders and generator
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
    with dnnlib.util.open_url(id_encoder) as f:
        id_eder = legacy.load_network_pkl(f)['E'].to(device).eval() # type: ignore
    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * 2)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * 2)
    # with open('/home/huangzixiong/dataset/classnerf/EG3D_GEN_DATASET/EG3D_GEN_W_0.5/pose_labels.json', 'r') as f:
    # with open('/home/huangzixiong/dataset/classnerf/FFHQ_in_the_wlid/label/labels.json', 'r') as f:
    with open('/home/huangzixiong/dataset/classnerf/afhq_v2/train/label/labels.json', 'r') as f:
        labels = json.load(f)

    # label_index = os.path.basename(id_image).replace('.jpg','.json')
    label_index = os.path.basename(id_image).replace('.jpg','.png')
    label = torch.tensor(labels[label_index],device=device).type(dtype=torch.float32).unsqueeze(0)
    # get id feature and pose feature
    id_feature = id_eder(image)
    style = id_feature
    # create camera intrinsics
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)

    cam2world_pose_s = LookAtPoseSampler.sample(3.14/2, 3.14/2,\
                                                camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    c_s = torch.cat([cam2world_pose_s.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    ws = G.mapping(z=style, c=torch.zeros_like(c_s))
    frame_num = 200

    pitch_range = 0.3
    yaw_range = 0.7
    cam2world_pose_d = LookAtPoseSampler.sample(3.1415926/2 + yaw_range * np.sin(2 * 3.1415926), 3.1415926/2 + pitch_range * np.cos(2 * 3.1415926),\
                                                camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)

    c_d = torch.cat([cam2world_pose_d.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    output = G.synthesis(ws=ws, c=label, noise_mode='const', neural_rendering_resolution=64)
    
    img = (output['image'][0] * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze()
    img_raw = (output['image_raw'][0] * 127.5 + 128).clamp(0, 255).to(torch.uint8).squeeze()
    img_depth = [normalize_depth(i, [i.max(), i.min()]) for i in -output['image_depth']]
    img_depth = img_depth[0]
    img = img.permute(1, 2, 0).cpu().numpy()
    img_raw = img_raw.permute(1, 2, 0).cpu().numpy()
    imageio.imwrite(os.path.join(outdir, 'fake1.png'), img)
    imageio.imwrite(os.path.join(outdir, 'fake2.png'), img_raw)
    imageio.imwrite(os.path.join(outdir, 'fake2_depth.png'), img_depth)
    # video_depth.append_data(img_depth[:, :, 0])
    

if __name__ == "__main__":
    generate_talking_videos()