import torch
import dnnlib
import cv2
import json
import os
import legacy
from camera_utils import LookAtPoseSampler

device = torch.device('cuda', 0)
id_image = 'samples/66666.jpg'
id_images = cv2.imread(id_image)
id_images = cv2.cvtColor(id_images, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None, ...]
id_images = (torch.from_numpy(id_images).to(device) / 127.5) - 1


with dnnlib.util.open_url('experiments/ablation_studies/20230510_pzl009_512_5120_final/00000-ffhq-ffhq-gpus2-batch24-gamma5/network-G_ema-latest.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
with dnnlib.util.open_url('experiments/ablation_studies/20230510_pzl009_512_5120_final/00000-ffhq-ffhq-gpus2-batch24-gamma5/network-E-latest.pkl') as f:
    id_eder = legacy.load_network_pkl(f)['E'].to(device).eval() # type: ignore
G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * 2)
G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * 2)

# get id feature and pose feature
id_feature = id_eder(id_images)
with open('/home/huangzixiong/dataset/classnerf/FFHQ_in_the_wlid/label/labels.json', 'r') as f:
    labels = json.load(f)
label_index = os.path.basename(id_image).replace('.jpg', '.png')
label = torch.tensor(labels[label_index],device=device).type(dtype=torch.float32).unsqueeze(0)
url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
with dnnlib.util.open_url(url) as f:
    vgg16 = torch.jit.load(f).eval().cuda()
target = cv2.imread(id_image)
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
target = torch.tensor(target, device=device).to(torch.float32)
target_images = target.unsqueeze(0).cuda().to(torch.float32)

# create camera intrinsics
intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
cam2world_pose_s = LookAtPoseSampler.sample(3.14/2, 3.14/2,\
                                            camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
c_s = torch.cat([cam2world_pose_s.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
c_s = c_s.repeat(id_feature.shape[0], 1)
ws = G.mapping(z=id_feature, c=torch.zeros_like(c_s))
output = G.synthesis(ws=ws, c=label, noise_mode='const', neural_rendering_resolution=64)

target_features = vgg16((id_images+1)*127.5, resize_images=True, return_lpips=True)
synth_features = vgg16((output['image'] + 1)*127.5, resize_images=True, return_lpips=True)
dist = (target_features - synth_features).square().sum()
img = (output['image'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
img = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)
cv2.imwrite("test.png", img)
print(dist.item())