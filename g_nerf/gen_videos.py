import os
import cv2
import json
import click
import dnnlib
import imageio
import numpy as np
import torch
from tqdm import tqdm
import math
import legacy
import time
import mrcfile
from camera_utils import LookAtPoseSampler
import random

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

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
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--prepared', 'prepared', help='whether to use image prepared in a folder')
@click.option('--id_image', 'id_image', help='Identity reference', required=True)
@click.option('--gen_shapes', 'gen_shapes', help='Generate mrcfile', type=bool, default=False,)
@click.option('--id_encoder', help='id_encoder', default='checkpoint/network-snapshot-000720.pkl')
@click.option('--gpu', type=int, help='gpu', default=0)
@click.option('--video_out_path', help='Output directory', type=str, default='video_results/', metavar='DIR')
@click.option('--outdir', help='Output directory', type=str, default='video_results/', metavar='DIR')
@click.option('--label_path', help='label_path', type=str, required=False, metavar='DIR')
@click.option('--res', help='Output directory', type=int, default=64, metavar='DIR')
@click.option('--dataset', help='Output directory', type=str, default='ffhq', metavar='DIR')
def generate_talking_videos(
    network_pkl: str,
    id_image: str,
    id_encoder: str,
    gpu: int, 
    video_out_path: str,
    outdir: str,
    prepared: str,
    res: int,
    gen_shapes: bool,
    label_path: str,
    dataset: str,
):
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)

    
    if prepared is not None:
        print('Loading image from "%s"...' % prepared)
        id_images = []
        id_images_path = os.listdir(prepared)
        id_images_path = [os.path.join(prepared, id_image) for id_image in id_images_path if id_image.endswith('.jpg')]
        for path in id_images_path:
            tmp_image = cv2.imread(path)
            tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None, ...]
            id_images.append(tmp_image)
        id_images = np.concatenate(id_images, axis=0)
        
    else:
        print('Loading image from "%s"...' % id_image)
        id_images = cv2.imread(id_image)
        id_images = cv2.cvtColor(id_images, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None, ...]

    gt_images = np.concatenate([i.transpose(1,2,0) for i in id_images], axis=1)
    scale = int(512 / res)
    gt_images_low = cv2.resize(gt_images, (int(gt_images.shape[1] / scale), res), interpolation=cv2.INTER_AREA)
    id_images = (torch.from_numpy(id_images).to(device) / 127.5) - 1

    # load encoders and generator
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
    with dnnlib.util.open_url(id_encoder) as f:
        id_eder = legacy.load_network_pkl(f)['E'].to(device).eval() # type: ignore
    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * 2)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * 2)

    # get id feature
    id_feature = id_eder(id_images)

    # create camera intrinsics
    if dataset == 'ffhq' or dataset == 'celeba':
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    elif dataset == 'shapenet':
        intrinsics = torch.tensor([[1.025390625, 0, 0.5], [0, 1.025390625, 0.5], [0, 0, 1]], device=device)

    if prepared is None:
        dir_name = os.path.basename(id_image).split('.')[0]
    else:
        dir_name = os.path.basename(prepared)
    if not os.path.exists(video_out_path):
        os.makedirs(video_out_path)
    video_raw = imageio.get_writer(os.path.join(video_out_path, dir_name+'_raw'+'.mp4'), mode='I', fps=30, codec='libx264')
    video_out = imageio.get_writer(os.path.join(video_out_path, dir_name+'.mp4'), mode='I', fps=30, codec='libx264')
    cam2world_pose_s = LookAtPoseSampler.sample(3.14/2, 3.14/2, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    c_s = torch.cat([cam2world_pose_s.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c_s = c_s.repeat(id_feature.shape[0], 1)
    ws = G.mapping(z=id_feature, c=torch.zeros_like(c_s))
    frame_num = 120


    for i in tqdm(range(0, frame_num)):
        pitch_range = 0.3
        yaw_range = 0.7
        cam2world_pose_d = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * i / frame_num), 3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * i / frame_num),\
                                                   radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        if dataset == 'shapenet':
            yaw_range = math.pi*2
            yaw = yaw_range * i / (frame_num -1)
            pitch_range = math.pi/4
            pitch = math.pi/3
            if 'cars' in id_image:
                radius = 1.3
            else:
                radius = 2.0
            cam2world_pose_d = LookAtPoseSampler.sample_srn(yaw, pitch, radius=radius, device=device)
    
        c_d = torch.cat([cam2world_pose_d.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).repeat(id_feature.shape[0], 1)
        output = G.synthesis(ws=ws, c=c_d, noise_mode='const', neural_rendering_resolution=res)

        img = (output['image'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        img_raw = (output['image_raw'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        img = flat_batch(img)
        img_raw = flat_batch(img_raw)

        img_depth = [normalize_depth(i, [i.max(), i.min()]) for i in -output['image_depth']]
        img_depth = flat_batch(img_depth)
        img_depth = cv2.cvtColor(img_depth, cv2.COLOR_GRAY2RGB)

        video_out.append_data(img)
        video_raw.append_data(img_raw)

        out_img_path = os.path.join(outdir, dir_name ,str(i) + '.png')
        if os.path.exists(os.path.dirname(out_img_path)) == False:
            os.makedirs(os.path.dirname(out_img_path))

    if gen_shapes:
        voxel_resolution=512
        max_batch = 10000000
        # generate shapes
        print('Generating shape')

        samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'])
        samples = samples.to(device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        with tqdm(total = samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = G.sample_mixed(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], ws, noise_mode='const')['sigma']
                    sigmas[:, head:head+max_batch] = sigma
                    head += max_batch
                    pbar.update(max_batch)

        sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
        sigmas = np.flip(sigmas, 0)

        pad = int(30 * voxel_resolution / 256)
        pad_top = int(38 * voxel_resolution / 256)
        sigmas[:pad] = 0
        sigmas[-pad:] = 0
        sigmas[:, :pad] = 0
        sigmas[:, -pad_top:] = 0
        sigmas[:, :, :pad] = 0
        sigmas[:, :, -pad:] = 0
        out_img_path = os.path.join(outdir, dir_name ,str(i) + '.png')
        with mrcfile.new_mmap(out_img_path.replace('png', 'mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
            mrc.data[:] = sigmas

if __name__ == "__main__":
    generate_talking_videos()