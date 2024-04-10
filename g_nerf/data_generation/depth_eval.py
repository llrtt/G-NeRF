import os
import glob
import sys
sys.path.append('talkingnerf')
import cv2
import json
import dnnlib
import numpy as np
import torch
import math
from tqdm import tqdm

import legacy

from camera_utils import LookAtPoseSampler
import aggregate_data
#----------------------------------------------------------------------------

def generate_random_samples(batch_size, G, all_label_paths, device):
    output_data = {}
    all_gen_z = torch.randn([batch_size, G.z_dim], device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
    all_poses_keys = list(all_label_paths.keys())
    all_poses_keys.sort()
    # range of view angles
    pitch_range = 0.3
    yaw_range = 0.55
    id_c = torch.randn([batch_size, 25], device=device)
    origin_c = torch.randn([batch_size, 25], device=device)
    loss_c = torch.randn([batch_size, 25], device=device)
    for i in range(batch_size):
        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * np.random.random()), 3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * np.random.random()),\
                                                    camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
        origin_c[i] = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * np.random.random()), 3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * np.random.random()),\
                                                    camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device) 
        loss_c[i] = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        random_indx = np.random.randint(0, len(all_poses_keys) - 1)
        id_c[i] = torch.tensor(all_label_paths[all_poses_keys[random_indx]], device=device)

    w_c = G.mapping(z=all_gen_z, c=id_c, truncation_psi=0.7, truncation_cutoff=14)
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

def generate_ffhq_samples(batch_size, G, all_label_paths, device):
    output_data = {}
    all_gen_z = torch.randn([batch_size, G.z_dim], device=device)
    all_poses_keys = list(all_label_paths.keys())
    all_poses_keys.sort()
    # range of view angles
    origin_c = torch.randn([batch_size, 25], device=device)
    loss_c = torch.randn([batch_size, 25], device=device)
    cam2world_pose_d = LookAtPoseSampler.sample(math.pi/2 - 0.5, math.pi/2, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    label = torch.cat([cam2world_pose_d.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    for i in range(batch_size):
        random_indx = np.random.randint(0, len(all_poses_keys) - 1)
        origin_c[i] = label
        # origin_c[i] = torch.tensor(all_label_paths[all_poses_keys[random_indx]], device=device)
            # for n in [3,7,11]:
            #     origin_c[i][n] = origin_c[i][n] / 2.7 * 3.2

        random_indx = np.random.randint(0, len(all_label_paths) - 1)
        loss_c[i] = torch.tensor(all_label_paths[all_poses_keys[random_indx]], device=device)
            # for n in [3,7,11]:
            #     loss_c[i][n] = loss_c[i][n] / 2.7 * 2.7

    w_c = G.mapping(z=all_gen_z, c=origin_c, truncation_psi=0.5, truncation_cutoff=14)
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
    output_data = {}
    all_gen_z = condition_z
    all_poses_keys = list(all_label_paths.keys())
    all_poses_keys.sort()
    # range of view angles
    origin_c = torch.randn([batch_size, 25], device=device)
    loss_c = torch.randn([batch_size, 25], device=device)
    cam2world_pose_d = LookAtPoseSampler.sample(math.pi/2 - 0.5, math.pi/2, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    label = torch.cat([cam2world_pose_d.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    for i in range(batch_size):
        random_indx = np.random.randint(0, len(all_poses_keys) - 1)
        origin_c[i] = label
        # origin_c[i] = torch.tensor(all_label_paths[all_poses_keys[random_indx]], device=device)
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
    root_dir = '/data_local/dataset/EG3D_GEN_DEPTH_EVAL'
    # FFHQ_images_dir = '/mnt/cephfs/dataset/Face/FFHQ_in_the_wlid/cropped_image'
    # all_images_paths = glob.glob(os.path.join(FFHQ_images_dir, '*.jpg'))
    # all_images_paths.sort()
    with open('/data_local/dataset/FFHQ_in_the_wlid/label/labels.json', 'r') as file:
        all_label_paths = json.load(file)
    image_number = 2000
    device = torch.device('cuda', 0)
    with dnnlib.util.open_url('ffhqrebalanced512-64.pkl') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * 2)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * 2)
    # with dnnlib.util.open_url('experiments/20230310_gpu018_train_encoder_512_fix_gen_ssim/00010-ffhq-eg3d-gpus4-batch8-gamma5/network-E-000300.pkl') as f:
    #     id_eder = legacy.load_network_pkl(f)['E'].to(device).eval() # type: ignore
    for num in tqdm(range(image_number)):

        # image = cv2.imread(all_images_paths[num])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        # image = ((torch.from_numpy(image)).to(device) / 255.0).unsqueeze(0) * 2 - 1

        # id_feature = id_eder(image)
        # style = id_feature

        out_dir = os.path.join(root_dir)

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        
        output_data = generate_ffhq_samples(1, G, all_label_paths, device)

        rgb_name_1 = os.path.join(out_dir, str(num).zfill(6) + '_f.jpg')
        rgb_name_2 = os.path.join(out_dir, str(num).zfill(6) + '_s.jpg')
        depth_name_1 = os.path.join(out_dir, str(num).zfill(6) + '_f.npy')
        depth_name_2 = os.path.join(out_dir, str(num).zfill(6) + '_s.npy')
        camera_name_1 = os.path.join(out_dir, str(num).zfill(6) + '_f.json')
        camera_name_2 = os.path.join(out_dir, str(num).zfill(6) + '_s.json')

        rgb_1 = ((output_data['origin_img'][0].cpu() + 1) * 127.5).numpy()
        rgb_2 = ((output_data['loss_img'][0].cpu() + 1) * 127.5).numpy()
        rgb_1 = np.rint(rgb_1).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        rgb_2 = np.rint(rgb_2).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)

        rgb_1 = cv2.cvtColor(rgb_1, cv2.COLOR_BGR2RGB)
        rgb_2 = cv2.cvtColor(rgb_2, cv2.COLOR_BGR2RGB)
        cv2.imwrite(rgb_name_1, rgb_1)
        # cv2.imwrite(rgb_name_2, rgb_2)

        # for name, data in zip([camera_name_1, camera_name_2], [output_data['origin_c'][0].cpu().numpy().tolist(), output_data['loss_c'][0].cpu().numpy().tolist()]):
        #     with open(name, 'w') as file:
        #         json.dump({os.path.basename(name): data}, file)

        for name, data in zip([depth_name_1], [output_data['origin_img_depth'][0].cpu().numpy(), output_data['loss_img_depth'][0].cpu().numpy()]):
            np.save(name, data)