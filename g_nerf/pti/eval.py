import cv2
import os
import glob
import json
import click
import sys
sys.path.append('talkingnerf')
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
import math
from tqdm import tqdm
from torchvision.transforms import Resize
import legacy
import skimage
from insightface.app import FaceAnalysis 

from camera_utils import LookAtPoseSampler
from gen_talking_videos import normalize_depth
from training.training_loop import load_pretrained
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
#----------------------------------------------------------------------------
def evalute_ssim(img_1, img_2) -> float:
    ssim_val = ssim(img_1 * 0.5 + 0.5, img_2 * 0.5 + 0.5, data_range=1.0, size_average=False) # return (N,)
    return ssim_val.mean().item()

def PSNR(img1, img2):
    img1 = img1 * 0.5 + 0.5
    img2 = img2 * 0.5 + 0.5
    # img1 and img2 have range [0, 1]
    mse = np.mean((img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

class LPIPS:
    def __init__(self, device) -> None:
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            self.vgg16 = torch.jit.load(f, map_location='cpu').eval().to(device)

    def cal(self, img1, img2):
        target_features = self.vgg16((img1 + 1) * 255 * 0.5, resize_images=True, return_lpips=True)
        synth_features = self.vgg16((img2 + 1) * 255 * 0.5, resize_images=True, return_lpips=True)
        p_loss = (target_features - synth_features).square().sum(1)
        return p_loss.mean().item()

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='experiments/20230428_pzl009_512_resolution_5120_small_pose/00000-ffhq-ffhq-gpus2-batch20-gamma5/network-G_ema-3300.pkl')
@click.option('--dataset', 'dataset', type=click.Choice(['ffhq', 'afhqv2', 'celeba']), help='whether to use image prepared in a folder', default='ffhq')
@click.option('--id_encoder', help='id_encoder', default='experiments/20230428_pzl009_512_resolution_5120_small_pose/00000-ffhq-ffhq-gpus2-batch20-gamma5/network-E-3300.pkl')
@click.option('--outdir', 'outdir', help='Identity reference', default='evaluation_results/ffhq')
@click.option('--dataset_dir', help='id_encoder', default='/home/huangzixiong/dataset/classnerf/FFHQ_in_the_wlid/cropped_image/')
@click.option('--res', help='Output directory', type=int, default=64, metavar='DIR')
@click.option('--gpu', type=int, help='gpu', default=0)
@click.option('--label_path', help='label_path', default='/home/huangzixiong/dataset/classnerf/FFHQ_in_the_wlid/label/labels.json')
@click.option('--isRand', 'isRand', help='isRand', default=True)
@click.option('--side_face', 'side_face', help='side_face', default=False)
@click.option('--only_depth', 'only_depth', help='isRand', default=False)
def generate_talking_videos(
    network_pkl: str,
    dataset: bool,
    outdir: str,
    dataset_dir: str, 
    id_encoder: str,
    res: int,
    gpu: int,
    label_path: str,
    isRand: bool,
    side_face: bool,
    only_depth: bool,
):
    device = torch.device('cuda', gpu)
    torch.cuda.set_device(device)
    lpips = LPIPS(device)
    # torch_resize = Resize([224,224])
    torch_resize_64 = Resize([res,res], antialias=True)
    # get id feature and pose feature

    # create camera intrinsics
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)

    if dataset == 'ffhq':
        all_paths = glob.glob(os.path.join(dataset_dir, "*.jpg"))
        all_paths.sort()
        l2_loss = torch.nn.MSELoss()
        all_paths.reverse()
        eval_imgs = all_paths[:8000]
        all_depths = glob.glob("/data_local/dataset/FFHQ_in_the_wlid/eval/*_depth.jpg")
        all_depths.sort()
        all_depths.reverse()
        with open(label_path, 'r') as f:
            labels = json.load(f)

    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(128, 128))

    avg_ssim_raw = 0
    avg_ssim_512 = 0

    avg_psnr_raw = 0
    avg_psnr_512 = 0

    avg_lpips_raw = 0
    avg_lpips_512 = 0

    avg_id_raw = 0
    avg_id_512 = 0

    avg_depth_64 = 0
    counter = 0
    outdir_fake_raw = os.path.join(outdir, 'fake', str(res))
    outdir_real_raw = os.path.join(outdir, 'real', str(res))
    outdir_real_512 = os.path.join(outdir, 'real', '512')
    outdir_fake_512 = os.path.join(outdir, 'fake', '512')
    outdir_depth = os.path.join(outdir, 'depth')
    for dir in [outdir_fake_raw, outdir_real_raw, outdir_real_512, outdir_fake_512, outdir_depth]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    for i,path in enumerate(tqdm(eval_imgs)):
        img_basename = os.path.basename(path)
        network_pkl = os.path.join('/data/baowenjie/20231115_backup/TalkingNeRF/pti_result/checkpoint', img_basename.replace('.jpg', '.pkl'))
        w_path = os.path.join('/data/baowenjie/20231115_backup/TalkingNeRF/pti_result/latent_code', img_basename.replace('.jpg', '.npy'))
        # load w and generator
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device).eval() # type: ignore
        ws = torch.from_numpy(np.load(w_path)).to(device)

        label_index = os.path.basename(path).replace('.jpg', '.png')
        if isRand:
            label_index = os.path.basename(all_paths[np.random.randint(0, len(all_paths))]).replace('.jpg', '.png')
        if dataset == 'celeba':
            label_index = label_index.replace('.png', '')
        label = torch.tensor(labels[label_index],device=device).type(dtype=torch.float32).unsqueeze(0)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        s_image = torch_resize_64((torch.from_numpy(image).to(device).to(torch.float32).unsqueeze(0) / 127.5 - 1))
        image = ((torch.from_numpy(image)).to(device) / 255.0).unsqueeze(0) * 2 - 1

        if side_face:
            cam2world_pose_d = LookAtPoseSampler.sample(math.pi/2 - 0.5, math.pi/2, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
            label = torch.cat([cam2world_pose_d.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        if only_depth:
            output = G.synthesis(ws=ws, c=label, noise_mode='const', neural_rendering_resolution=res, only_depth=only_depth)
        else:
            output = G.synthesis(ws=ws, c=label, noise_mode='const', neural_rendering_resolution=res)

            # ssim_raw = evalute_ssim(output['image_raw'], s_image)
            # ssim_512 = evalute_ssim(output['image'].cpu(), image.cpu())
            ssim_raw = skimage.metrics.structural_similarity(np.clip(output['image_raw'].squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1), np.clip(s_image.squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1),data_range=1, multichannel=True, channel_axis=2)
            ssim_512 = skimage.metrics.structural_similarity(np.clip(output['image'].squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1), np.clip(image.squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1),data_range=1, multichannel=True, channel_axis=2)
            avg_ssim_raw = avg_ssim_raw * i / (i+1) + ssim_raw / (i+1)
            avg_ssim_512 = avg_ssim_512 * i / (i+1) + ssim_512 / (i+1)
            
            psnr_raw = skimage.metrics.peak_signal_noise_ratio(np.clip(output['image_raw'].squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1), np.clip(s_image.squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1),data_range=1)
            psnr_512 = skimage.metrics.peak_signal_noise_ratio(np.clip(output['image'].squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1), np.clip(image.squeeze().permute(1,2,0).cpu().numpy()*0.5+0.5,0,1),data_range=1)
            avg_psnr_raw = avg_psnr_raw * i / (i+1) + psnr_raw / (i+1)
            avg_psnr_512 = avg_psnr_512 * i / (i+1) + psnr_512 / (i+1)

            lpips_raw = lpips.cal(output['image_raw'], s_image)
            lpips_512 = lpips.cal(output['image'], image)
            avg_lpips_raw = avg_lpips_raw * i / (i+1) + lpips_raw / (i+1)
            avg_lpips_512 = avg_lpips_512 * i / (i+1) + lpips_512 / (i+1)

            fake_raw = output['image_raw'].squeeze().permute(1,2,0).cpu().numpy()
            fake_raw = np.clip(fake_raw * 127.5 + 128, 0, 255).astype(np.uint8)
            fake_raw = cv2.cvtColor(fake_raw, cv2.COLOR_RGB2BGR)
            fake_512 = output['image'].squeeze().permute(1,2,0).cpu().numpy()
            fake_512 = np.clip(fake_512 * 127.5 + 128, 0, 255).astype(np.uint8)
            fake_512 = cv2.cvtColor(fake_512, cv2.COLOR_RGB2BGR)
            real_raw = s_image.squeeze().permute(1,2,0).cpu().numpy()
            real_raw = np.clip(real_raw * 127.5 + 128, 0, 255).astype(np.uint8)
            real_raw = cv2.cvtColor(real_raw, cv2.COLOR_RGB2BGR)
            real_512 = image.squeeze().permute(1,2,0).cpu().numpy()
            real_512 = np.clip(real_512 * 127.5 + 128, 0, 255).astype(np.uint8)
            real_512 = cv2.cvtColor(real_512, cv2.COLOR_RGB2BGR)

            real_embedding_raw = app.get(real_raw)
            fake_embedding_raw = app.get(fake_raw)
            real_embedding_512 = app.get(real_512)
            fake_embedding_512 = app.get(fake_512)
            if len(real_embedding_raw) == 0 or len(fake_embedding_raw) == 0 or len(real_embedding_512) == 0 or len(fake_embedding_512) == 0:
                pass
            else:
                real_embedding_raw = real_embedding_raw[0].embedding
                fake_embedding_raw = fake_embedding_raw[0].embedding

                real_embedding_512 = real_embedding_512[0].embedding
                fake_embedding_512 = fake_embedding_512[0].embedding
                id_raw = np.dot(real_embedding_raw, fake_embedding_raw) / (np.linalg.norm(real_embedding_raw) * np.linalg.norm(fake_embedding_raw))
                id_512 = np.dot(real_embedding_512, fake_embedding_512) / (np.linalg.norm(real_embedding_512) * np.linalg.norm(fake_embedding_512))
                avg_id_raw = avg_id_raw * counter / (counter+1) + id_raw / (counter+1)
                avg_id_512 = avg_id_512 * counter / (counter+1) + id_512 / (counter+1)
                counter += 1
                print(f'avg_id_raw: {avg_id_raw}, avg_id_512: {avg_id_512}')
        if dataset == 'ffhq' or dataset == 'celeba':
            depth_path = os.path.join(os.path.dirname(all_depths[i]), os.path.basename(eval_imgs[i]).replace('.jpg', '_depth.jpg'))
            depth = cv2.imread(depth_path, 0)
            mask = cv2.imread(depth_path.replace('depth.', 'mask.'), 0)
            depth = cv2.resize(depth, (res,res)).astype(np.float32)
            mask = cv2.resize(mask, (res,res))
            #normalize
            depth[mask > 0] = (depth[mask > 0] - depth[mask > 0].mean()) / depth.std()
            tmp_mask = np.expand_dims(np.expand_dims(mask, 0), 0)
            tmp_mask = torch.from_numpy(tmp_mask)
            fake_depth = output['image_depth']
            fake_depth = (fake_depth - (fake_depth)[tmp_mask > 0].min()).clamp(0,255)
            fake_depth[tmp_mask > 0] = (1 - fake_depth[tmp_mask > 0] / fake_depth[tmp_mask > 0].max()) * 255
            fake_depth[tmp_mask > 0] = (fake_depth[tmp_mask > 0] - fake_depth[tmp_mask > 0].mean()) / fake_depth[tmp_mask > 0].std()
            depth_loss = l2_loss(fake_depth[tmp_mask > 0], torch.from_numpy(np.expand_dims(np.expand_dims(depth, 0), 0))[tmp_mask > 0].to(fake_depth.device))
            avg_depth_64 = avg_depth_64 * i / (i+1) + depth_loss / (i+1)
            print(f'avg_ssim_raw: {avg_ssim_raw}, avg_ssim_512: {avg_ssim_512}, avg_psnr_raw: {avg_psnr_raw}, avg_psnr_512: {avg_psnr_512}, avg_lpips_raw: {avg_lpips_raw}, avg_lpips_512: {avg_lpips_512}, avg_id_raw: {avg_id_raw}, avg_id_512: {avg_id_512}, avg_depth_64: {avg_depth_64}')
        else:
            print(f'avg_ssim_raw: {avg_ssim_raw}, avg_ssim_512: {avg_ssim_512}, avg_psnr_raw: {avg_psnr_raw}, avg_psnr_512: {avg_psnr_512}, avg_lpips_raw: {avg_lpips_raw}, avg_lpips_512: {avg_lpips_512}')

        if not only_depth:
            img = (output['image_raw'][0] * 127.5 + 128).permute(1,2,0).cpu().clamp(0, 255).numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            s_image = (s_image.squeeze(0) * 127.5 + 128).squeeze().permute(1,2,0).cpu().clamp(0, 255).numpy().astype(np.uint8)
            s_image = cv2.cvtColor(s_image, cv2.COLOR_RGB2BGR)
            label_index = label_index.replace('.png', '')
            cv2.imwrite(os.path.join(outdir_fake_raw, os.path.basename(path)), img)
            cv2.imwrite(os.path.join(outdir_real_raw, os.path.basename(path)), s_image)

            img = (output['image'][0] * 127.5 + 128).permute(1,2,0).cpu().clamp(0, 255).numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image = (image.squeeze(0) * 127.5 + 128).squeeze().permute(1,2,0).cpu().clamp(0, 255).numpy().astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(outdir_fake_512, os.path.basename(path)), img)
            cv2.imwrite(os.path.join(outdir_real_512, os.path.basename(path)), image)
        img_depth = [normalize_depth(i, [i.max(), i.min()]) for i in -output['image_depth']]
        cv2.imwrite(os.path.join(outdir_depth, os.path.basename(path)), img_depth[0])
    with open(os.path.join(outdir, 'evaluation.txt'), 'w') as f:
        f.write(f'avg_ssim_raw: {avg_ssim_raw}, avg_ssim_512: {avg_ssim_512}, avg_psnr_raw: {avg_psnr_raw}, avg_psnr_512: {avg_psnr_512}, avg_lpips_raw: {avg_lpips_raw}, avg_lpips_512: {avg_lpips_512}, avg_id_raw: {avg_id_raw}, avg_id_512: {avg_id_512}, avg_depth_64: {avg_depth_64}')

if __name__ == "__main__":
    generate_talking_videos()

