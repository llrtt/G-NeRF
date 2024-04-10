import os
import glob
import sys
sys.path.append('talkingnerf')
import cv2
import json
import dnnlib
import numpy as np
import torch
from tqdm import tqdm
import mrcfile
import legacy

from camera_utils import LookAtPoseSampler
#----------------------------------------------------------------------------
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

def generate_random_samples(batch_size, G, all_label_paths, device):
    output_data = {}
    all_gen_z = torch.randn([batch_size, G.z_dim], device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
    all_poses_keys = list(all_label_paths.keys())
    all_poses_keys.sort()
    # range of view angles
    pitch_range = 0.6
    yaw_range = 1.1
    id_c = torch.randn([batch_size, 25], device=device)
    origin_c = torch.randn([batch_size, 25], device=device)
    loss_c = torch.randn([batch_size, 25], device=device)
    for i in range(batch_size):
        cam2world_pose = LookAtPoseSampler.sample_origin(3.14/2 + yaw_range * np.sin(2 * 3.14 * np.random.random()), 3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * np.random.random()),\
                                                    camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        origin_c[i] = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        cam2world_pose = LookAtPoseSampler.sample_origin(3.14/2 + yaw_range * np.sin(2 * 3.14 * np.random.random()), 3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * np.random.random()),\
                                                    camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device) 
        loss_c[i] = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        random_indx = np.random.randint(0, len(all_poses_keys) - 1)
        id_c[i] = torch.tensor(all_label_paths[all_poses_keys[random_indx]], device=device)

    w_c = G.mapping(z=all_gen_z, c=id_c, truncation_psi=0.5, truncation_cutoff=None)
    origin_img = G.synthesis(ws=w_c, c=origin_c, noise_mode='const')
    # w_c = G.mapping(z=all_gen_z, c=loss_c)
    loss_img = G.synthesis(ws=w_c, c=loss_c, noise_mode='const')

    output_data['origin_img']       = origin_img['image']
    output_data['origin_img_depth'] = origin_img['image_depth']
    output_data['origin_c']         = origin_c
    output_data['loss_img']         = loss_img['image']
    output_data['loss_img_depth']   = loss_img['image_depth']
    output_data['loss_c']           = loss_c
    output_data['w_c']              = w_c
    return output_data

def generate_ffhq_samples(batch_size, G, device):
    with open('/home/huangzixiong/dataset/classnerf/FFHQ_in_the_wlid/label/labels.json', 'r') as file:
        all_label_paths = json.load(file)
    output_data = {}
    all_gen_z = torch.randn([batch_size, G.z_dim], device=device)
    all_poses_keys = list(all_label_paths.keys())
    all_poses_keys.sort()
    # range of view angles
    origin_c = torch.randn([batch_size, 25], device=device)
    loss_c = torch.randn([batch_size, 25], device=device)
    for i in range(batch_size):
        random_indx = np.random.randint(0, len(all_poses_keys) - 1)
        origin_c[i] = torch.tensor(all_label_paths[all_poses_keys[random_indx]], device=device)
            # for n in [3,7,11]:
            #     origin_c[i][n] = origin_c[i][n] / 2.7 * 3.2

        random_indx = np.random.randint(0, len(all_label_paths) - 1)
        loss_c[i] = torch.tensor(all_label_paths[all_poses_keys[random_indx]], device=device)
            # for n in [3,7,11]:
            #     loss_c[i][n] = loss_c[i][n] / 2.7 * 2.7

    w_c = G.mapping(z=all_gen_z, c=origin_c, truncation_psi=0.6)
    origin_img = G.synthesis(ws=w_c, c=origin_c, noise_mode='const')
    # w_c = G.mapping(z=all_gen_z, c=loss_c)
    loss_img = G.synthesis(ws=w_c, c=loss_c, noise_mode='const')

    output_data['origin_img']       = origin_img['image']
    output_data['origin_img_depth'] = origin_img['image_depth']
    output_data['origin_c']         = origin_c
    output_data['loss_img']         = loss_img['image']
    output_data['loss_img_depth']   = loss_img['image_depth']
    output_data['loss_c']           = loss_c
    return output_data


def generate_conditional_samples(batch_size, G, device, condition_z):
    with open('/home/huangzixiong/dataset/classnerf/FFHQ_in_the_wlid/label/labels.json', 'r') as file:
        all_label_paths = json.load(file)
    output_data = {}
    all_gen_z = condition_z
    all_poses_keys = list(all_label_paths.keys())
    all_poses_keys.sort()
    # range of view angles
    origin_c = torch.randn([batch_size, 25], device=device)
    loss_c = torch.randn([batch_size, 25], device=device)
    for i in range(batch_size):
        random_indx = np.random.randint(0, len(all_poses_keys) - 1)
        origin_c[i] = torch.tensor(all_label_paths[all_poses_keys[random_indx]], device=device)
            # for n in [3,7,11]:
            #     origin_c[i][n] = origin_c[i][n] / 2.7 * 3.2

        random_indx = np.random.randint(0, len(all_label_paths) - 1)
        loss_c[i] = torch.tensor(all_label_paths[all_poses_keys[random_indx]], device=device)
            # for n in [3,7,11]:
            #     loss_c[i][n] = loss_c[i][n] / 2.7 * 2.7

    w_c = G.mapping(z=all_gen_z, c=torch.zeros_like(origin_c))
    origin_img = G.synthesis(ws=w_c, c=origin_c, noise_mode='const', neural_rendering_resolution=128)
    # w_c = G.mapping(z=all_gen_z, c=loss_c)
    loss_img = G.synthesis(ws=w_c, c=loss_c, noise_mode='const', neural_rendering_resolution=128)

    output_data['origin_img']       = origin_img['image']
    output_data['origin_img_depth'] = origin_img['image_depth']
    output_data['origin_c']         = origin_c
    output_data['loss_img']         = loss_img['image']
    output_data['loss_img_depth']   = loss_img['image_depth']
    output_data['loss_c']           = loss_c
    return output_data

def generate_random_pose_conditional_samples(batch_size, G, device, condition_z):
    output_data = {}
    all_gen_z = condition_z
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)

    # range of view angles
    pitch_range = 1
    yaw_range = 1.4
    origin_c = torch.randn([batch_size, 25], device=device)
    loss_c = torch.randn([batch_size, 25], device=device)
    for i in range(batch_size):
        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * np.random.random()), 3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * np.random.random()),\
                                                    camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        origin_c[i] = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * np.random.random()), 3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * np.random.random()),\
                                                    camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device) 
        loss_c[i] = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    w_c = G.mapping(z=all_gen_z, c=torch.zeros_like(origin_c))
    origin_img = G.synthesis(ws=w_c, c=origin_c, noise_mode='const', neural_rendering_resolution=128)
    # w_c = G.mapping(z=all_gen_z, c=loss_c)
    loss_img = G.synthesis(ws=w_c, c=loss_c, noise_mode='const', neural_rendering_resolution=128)

    output_data['origin_img']       = origin_img['image']
    output_data['origin_img_depth'] = origin_img['image_depth']
    output_data['origin_c']         = origin_c
    output_data['loss_img']         = loss_img['image']
    output_data['loss_img_depth']   = loss_img['image_depth']
    output_data['loss_c']           = loss_c
    return output_data


    

if __name__ == "__main__":
    gen_shapes = False
    root_dir = '/data_14t/data_hzx/zs3nerf/evaluation_results/truncation_effect'
    with open('/data_14t/data_hzx/zs3nerf/FFHQ_in_the_wlid/label/labels.json', 'r') as file:
        all_label_paths = json.load(file)
    image_number = 3
    device = torch.device('cuda', 0)
    with dnnlib.util.open_url('ffhq512-128.pkl') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * 2)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * 2)
    batch_size = 1

    for num in tqdm(range(image_number)):

        output_data = {}
        all_gen_z = torch.randn([batch_size, G.z_dim], device=device)
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
        camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
        all_poses_keys = list(all_label_paths.keys())
        all_poses_keys.sort()
        # range of view angles
        pitch_range = 0.6
        yaw_range = 1.1
        id_c = torch.randn([batch_size, 25], device=device)
        origin_c = torch.randn([batch_size, 25], device=device)

        for i in range(batch_size):
            random_indx = np.random.randint(0, len(all_poses_keys) - 1)
            origin_c[i] = torch.tensor(all_label_paths[all_poses_keys[random_indx]], device=device)
            random_indx = np.random.randint(0, len(all_poses_keys) - 1)
            id_c[i] = torch.tensor(all_label_paths[all_poses_keys[random_indx]], device=device)

        for t in [0, 0.3, 0.5, 0.7, 1.0]:
            w_c = G.mapping(z=all_gen_z, c=id_c, truncation_psi=t, truncation_cutoff=14)
            origin_img = G.synthesis(ws=w_c, c=origin_c, noise_mode='const')

            out_dir = os.path.join(root_dir, 'truncation_' + str(t).replace('.', '_'))

            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            
            output_data = generate_random_samples(1, G, all_label_paths, device)

            rgb_name = os.path.join(out_dir, 'id_' + str(num) + '_' + str(t) + '.jpg')
            rgb_raw_name = os.path.join(out_dir, 'id_' + str(num) + '_' + str(t) + '_raw.jpg')
            depth_name = os.path.join(out_dir, 'id_' + str(num) + '_' + str(t) + '_depth.jpg')

            rgb = ((origin_img['image'][0].cpu() + 1) * 127.5).numpy()
            rgb_raw = ((origin_img['image_raw'][0].cpu() + 1) * 127.5).numpy()
            rgb = np.rint(rgb).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
            rgb_raw = np.rint(rgb_raw).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb_raw = cv2.cvtColor(rgb_raw, cv2.COLOR_BGR2RGB)

            depth_image = -origin_img['image_depth'][0].cpu().numpy()
            depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
            depth_image = (depth_image * 255).astype(np.uint8).transpose(1, 2, 0)
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(rgb_name, rgb)
            cv2.imwrite(rgb_raw_name, rgb_raw)
            cv2.imwrite(depth_name, depth_image)
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
                ws = output_data['w_c']
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
                with mrcfile.new_mmap(rgb_name.replace('png', 'mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas