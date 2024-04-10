import os
import cv2
import json
import click
import sys
sys.path.append('talkingnerf')
import dnnlib
import imageio
import numpy as np
import torch
from tqdm import tqdm
import math
import legacy
import time
from camera_utils import LookAtPoseSampler

def flat_batch(batch_image):
    flatten_image = [i for i in batch_image]
    return np.concatenate(flatten_image, axis=1)

def normalize_depth(depth, range):
    hi, lo = range
    depth = (depth - lo) * (255 / (hi - lo))
    return depth.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

def angleToArc(angle):
    return angle * math.pi / 180
#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='ffhq_pti.pkl')
@click.option('--w_path', 'w_path', help='Network pickle filename', default='evaluation_results/pti/66666.npy')
@click.option('--gpu', type=int, help='gpu', default=0)
@click.option('--outdir', help='Output directory', type=str, default='./', metavar='DIR')
def generate_talking_videos(
    network_pkl: str,
    outdir: str,
    w_path: str,
    gpu: int,
):
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)

    # load encoders and generator
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
    # G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * 2)
    # G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * 2)

    # create camera intrinsics
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)

    video_raw = imageio.get_writer('test_raw.mp4', mode='I', fps=30, codec='libx264')
    video_out = imageio.get_writer('test.mp4', mode='I', fps=30, codec='libx264')
    start = time.time()
    ws = torch.from_numpy(np.load(w_path)).to(device)
    frame_num = 7
    yaw_angles = [45,60,75,90,105,130,145]
    pitch_angles = [60,70,80,90,100,110,120]
    pitch_angles = [90,90,90,90,90,90,90]
    
    for i in tqdm(range(0, frame_num)):
        pitch_range = 0.45
        yaw_range = 0.7
        cam2world_pose_d = LookAtPoseSampler.sample(angleToArc(yaw_angles[i]), angleToArc(pitch_angles[i]),\
                                                     radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        # cam2world_pose_d = cam2world_pose_d @ torch.tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=device).float()
        c_d = torch.cat([cam2world_pose_d.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).repeat(ws.shape[0], 1)
        output = G.synthesis(ws=ws, c=c_d, noise_mode='const', neural_rendering_resolution=64)
        
        img = (output['image'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        img_raw = (output['image_raw'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        img = flat_batch(img)
        img_raw = flat_batch(img_raw)
        # img = np.concatenate([gt_images, img], 0)

        img_depth = [normalize_depth(i, [i.max(), i.min()]) for i in -output['image_depth']]
        img_depth = flat_batch(img_depth)
        img_depth = cv2.cvtColor(img_depth, cv2.COLOR_GRAY2RGB)
        img_raw = np.concatenate([img_raw, img_depth], 0)

        video_out.append_data(img)
        video_raw.append_data(img_raw)

        out_img_path = os.path.join('./evaluation_results/pti', os.path.basename(w_path).replace('.npy', '') ,str(i) + '.png')
        if os.path.exists(os.path.dirname(out_img_path)) == False:
            os.makedirs(os.path.dirname(out_img_path))
        cv2.imwrite(out_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print('total time: ', time.time() - start)

if __name__ == "__main__":
    generate_talking_videos()