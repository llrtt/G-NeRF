# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""
import os
import cv2
import random
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from torchvision.transforms import Resize
from training.dual_discriminator import filtered_resizing
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import legacy

def load_pretrained(class_name, path, device, pretrained=True, num_gpus=1):
    model = dnnlib.util.construct_class_by_name(class_name=class_name, num_gpus=num_gpus).requires_grad_(False).to(device) # subclass of torch.nn.Module
    if pretrained:
        with dnnlib.util.open_url(path) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('E', model)]:
            misc.copy_params_and_buffers(resume_data[name].eval(), module, require_all=False)
    return model.eval()  

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(768 // training_set.image_shape[2], 7, 32)
    gh = np.clip(432 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    # if not training_set.has_labels:
    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    images, labels = zip(*[(training_set[i]["loss_image"], training_set[i]["loss_c"]) for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').resize([gw * 64, gh * 64]).save(fname)

#----------------------------------------------------------------------------
# Kaiming initialization.
def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    testing_set_kwargs      = {},       # Options for testing set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    train_en                = False,    # Train encoder?
    train_gen               = False,    # Train generator?
    gan_depth               = True,     # GAN depth loss?
    update_model            = False,    # Update model?
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_gen              = None,     # generator pickle to resume training from.
    resume_en               = None,     # encoder pickle to resume training from.
    resume_disc             = None,     # discriminator pickle to resume training from. 
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    assert train_en or train_gen, "Nothing is to be trained! You need to train something!"
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    print('Loading testing set...')
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    # print('Loading testing set...')
    dataset_kwargs = dnnlib.EasyDict(**testing_set_kwargs)
    test_set = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()
    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    id_encoder = dnnlib.util.construct_class_by_name(class_name='training.networks_stylegan2.ResNeXt50', num_gpus=num_gpus, out_dim=G_kwargs.z_dim).requires_grad_(False).to(device) # subclass of torch.nn.Module
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G.register_buffer('dataset_label_std', training_set.get_label_std().clone().detach().to(device))
    id_encoder.apply(weights_init)
    G.apply(weights_init)
    
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f, map_location='cpu').eval().to(device)
    ssim_resize = Resize([loss_kwargs.neural_rendering_resolution_initial, loss_kwargs.neural_rendering_resolution_initial], antialias=True).to(device)

    if gan_depth:
        D = dnnlib.util.construct_class_by_name(**D_kwargs, **dict(c_dim=training_set.label_dim, img_resolution=loss_kwargs.neural_rendering_resolution_initial, img_channels=1)).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        D.apply(weights_init)
        # Resume from existing pickle.
        if (resume_disc is not None) and (rank == 0):
            print(f'Discriminator is resuming from "{resume_disc}"')
            with dnnlib.util.open_url(resume_disc) as f:
                resume_data = legacy.load_network_pkl(f)
            for name, module in [('D', D)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Resume from existing pickle.
    if (resume_gen is not None) and (rank == 0):
        print(f'Generator is resuming from "{resume_gen}"')
        with dnnlib.util.open_url(resume_gen) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G_ema', G)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
        if update_model:
            snapshot_data = {}
            module = copy.deepcopy(G).eval().requires_grad_(False).cpu()
            snapshot_data['G_ema'] = module
            del module # conserve memory
            snapshot_pkl = resume_gen.replace('.pkl', '_update.pkl')
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)

    if (resume_en is not None) and (rank == 0):
        print(f'Encoder is resuming from "{resume_en}"')
        with dnnlib.util.open_url(resume_en) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('E', id_encoder)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
        if update_model:
            snapshot_data = {}
            module = copy.deepcopy(id_encoder).eval().requires_grad_(False).cpu()
            snapshot_data['E'] = module
            del module # conserve memory
            snapshot_pkl = resume_en.replace('.pkl', '_update.pkl')
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)
            print('model updated! Exiting...')
            exit()

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    modules = [id_encoder, G]

    if gan_depth:
        opt_D = torch.optim.Adam(list(D.parameters()), D_opt_kwargs.lr, betas=[0,0.999], eps=1e-8)
        modules = [id_encoder, G, D]
    for module in modules:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None

    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=test_set)
        
        del test_set

        save_image_grid(images, os.path.join(run_dir, 'id_images.png'), drange=[0,255], grid_size=grid_size)

        real_raw_images = (ssim_resize(torch.from_numpy(images)).to(device) / 255.0) * 2 - 1
        images = ((torch.from_numpy(images)).to(device) / 255.0) * 2 - 1
        grid_c = torch.from_numpy(labels).to(device).split(1)

        grid_size, train_images, train_labels = setup_snapshot_image_grid(training_set=training_set)

        save_image_grid(train_images, os.path.join(run_dir, 'train_id_images.png'), drange=[0,255], grid_size=grid_size)

        train_images = ((torch.from_numpy(train_images)).to(device) / 255.0) * 2 - 1
        train_grid_c = torch.from_numpy(train_labels).to(device).split(1)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    param_list = []

    #warming
    all_gen_z = torch.zeros([batch_size//num_gpus, G_kwargs.z_dim], device=device)
    phase_real_c = torch.zeros([batch_size//num_gpus, 25], device=device)
    ws = G.mapping(all_gen_z, phase_real_c, update_emas=False)
    l1_gen_img = G.synthesis(ws, phase_real_c, neural_rendering_resolution=loss_kwargs.neural_rendering_resolution_initial, update_emas=False)
    torch.cuda.empty_cache()
    # Set training parameters, when z_dim is not equal to 512, we need to train the mapping net work.
    if train_en:
        id_encoder.train()
        param_list += list(id_encoder.parameters())
        if G.z_dim != 512 and not train_gen:
            print("Train encoder together with mapping network")
            param_list += list(G.backbone.mapping.parameters())
    if train_gen:
        param_list += list(G.parameters())
    
    opt = torch.optim.Adam(param_list, G_opt_kwargs.lr, betas=[0.9,0.999], eps=1e-8)
    L1_loss = torch.nn.L1Loss(reduction='none')
    max_ssim = -100.0
    
    while True:
        # Set training state
        opt.zero_grad(set_to_none=True)
        if train_en:
            id_encoder.requires_grad_(True)
            if G.z_dim != 512 and not train_gen:
                G.backbone.mapping.requires_grad_(True)
        if train_gen:
            G.requires_grad_(True)

        # Fetch training data.
        train_data = next(training_set_iterator)
        id_images = ((train_data["condition_image"]).to(device) / 255.0) * 2 - 1

        id_feature = id_encoder(id_images)
        style = id_feature
        phase_real_img = (train_data["loss_image"].to(device).to(torch.float32) / 127.5 - 1)
        phase_real_c = train_data["loss_c"].to(device)
        all_gen_z = style
        ws = G.mapping(all_gen_z, phase_real_c, update_emas=False)

        l1_gen_img = G.synthesis(ws, phase_real_c, neural_rendering_resolution=loss_kwargs.neural_rendering_resolution_initial, update_emas=False)

        real_img_raw = (ssim_resize(train_data["loss_image"]).to(device) / 255.0) * 2 - 1
        real_img = {'image': phase_real_img, 'image_raw': real_img_raw}

        # Reconstruction loss
        p_real = real_img['image_raw'].clone()
        p_gen = l1_gen_img['image_raw'].clone()
        l1_loss_raw = L1_loss(p_real, p_gen).mean((1,2,3))
        ssim_val_raw = 1-ssim(p_real * 0.5 + 0.5, p_gen * 0.5 + 0.5, data_range=1.0, size_average=False) # return (N,)
        target_features = vgg16((p_real + 1) * 255 * 0.5, resize_images=True, return_lpips=True)
        synth_features = vgg16((p_gen + 1) * 255 * 0.5, resize_images=True, return_lpips=True)
        p_loss_raw = (target_features - synth_features).square().sum(1)

        p_real = real_img['image'].clone()
        p_gen = l1_gen_img['image'].clone()
        l1_loss = L1_loss(p_real, p_gen).mean((1,2,3))
        target_features = vgg16((p_real + 1) * 255 * 0.5, resize_images=True, return_lpips=True)
        synth_features = vgg16((p_gen + 1) * 255 * 0.5, resize_images=True, return_lpips=True)
        p_loss = (target_features - synth_features).square().sum(1)
        l_ssim_val = 1-ssim(p_real * 0.5 + 0.5, p_gen * 0.5 + 0.5, data_range=1.0, size_average=False) # return (N,)

        # gan loss
        # gen_logits = D(l1_gen_img['image_depth'], phase_real_c)
        # loss_Gmain = torch.nn.functional.softplus(-gen_logits).mean()
        refine_factor = (train_data['factor'].sum() + 1e-6)
        training_stats.report('Loss/G/l1_loss', ((l1_loss)*train_data['factor'].to(device)).sum() / refine_factor)
        training_stats.report('Loss/G/l_ssim_val', ((l_ssim_val)*train_data['factor'].to(device)).sum() / refine_factor)
        training_stats.report('Loss/G/p_loss', ((p_loss)*train_data['factor'].to(device)).sum() / refine_factor)

        training_stats.report('Loss/G/l1_loss_raw', ((l1_loss_raw)*train_data['factor'].to(device)).sum() / refine_factor)
        training_stats.report('Loss/G/ssim_val_raw', ((ssim_val_raw)*train_data['factor'].to(device)).sum() / refine_factor)
        training_stats.report('Loss/G/p_loss_raw', ((p_loss_raw)*train_data['factor'].to(device)).sum() / refine_factor)
        
        # gan loss
        if gan_depth:
            gen_logits = D(l1_gen_img['image_depth'], phase_real_c)
            loss_Gmain = torch.nn.functional.softplus(-gen_logits).mean()
            training_stats.report('Loss/G/main', loss_Gmain)
            loss = (((l1_loss + l_ssim_val + p_loss + l1_loss_raw + ssim_val_raw +p_loss_raw)*train_data['factor'].to(device)).sum() / refine_factor + 1.2*loss_Gmain)
        else:
            loss = (((l1_loss + l_ssim_val + p_loss + l1_loss_raw + ssim_val_raw +p_loss_raw)*train_data['factor'].to(device)).sum() / refine_factor)
        loss.backward()
    

        param_grads = []
        if train_en:
            param_grads += [param for param in id_encoder.parameters() if param.numel() > 0 and param.grad is not None]
            if G.z_dim != 512 and not train_gen:
                param_grads += [param for param in G.backbone.mapping.parameters() if param.numel() > 0 and param.grad is not None]
        if train_gen:
           param_grads += [param for param in G.parameters() if param.numel() > 0 and param.grad is not None]

        if len(param_grads) > 0:
            flat = torch.cat([param.grad.flatten() for param in param_grads])
            if num_gpus > 1:
                torch.distributed.all_reduce(flat)
                flat /= num_gpus
            misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in param_grads])
            for param, grad in zip(param_grads, grads):
                param.grad = grad.reshape(param.shape)
        opt.step()
        id_encoder.requires_grad_(False)
        G.requires_grad_(False)

        # Dmain: Update discriminator.
        if gan_depth:       
            opt_D.zero_grad(set_to_none=True)
            D.requires_grad_(True)

            # Dmain: Minimize logits for generated images.
            gen_logits = D(l1_gen_img['image_depth'].detach(), phase_real_c)
            loss_Dgen = torch.nn.functional.softplus(gen_logits)
            training_stats.report('Loss/scores/fake', gen_logits)
            (loss_Dgen).mean().backward()

            # Dmain: Maximize logits for real images.
            # Dr1: Apply R1 regularization.
            c_depth_image = ssim_resize(train_data['c_depth_image']).to(device).detach().requires_grad_(True)
            real_logits = D(c_depth_image, train_data['condition_c'].to(device))
            loss_Dreal = torch.nn.functional.softplus(-real_logits)
            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[c_depth_image], create_graph=True, only_inputs=True)
            r1_grads_image = r1_grads[0]
            r1_penalty = r1_grads_image.square().sum([1, 2, 3])
            loss_Dr1 = r1_penalty * (loss_kwargs.r1_gamma / 2)
            (loss_Dreal + loss_Dr1).mean().backward()
            D.requires_grad_(False)

            training_stats.report('Loss/D/real', loss_Dreal.mean())
            training_stats.report('Loss/D/r1', loss_Dr1.mean())

            param_grads = []
            param_grads += D.parameters()
            flat = torch.cat([param.grad.flatten() for param in param_grads])
            if num_gpus > 1:
                torch.distributed.all_reduce(flat)
                flat /= num_gpus
            misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
            grads = flat.split([param.numel() for param in param_grads])
            for param, grad in zip(param_grads, grads):
                param.grad = grad.reshape(param.shape)
            opt_D.step()
        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        try:
            fields = []
            fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
            fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
            fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
            fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
            fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
            fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
            fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
            fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
            torch.cuda.reset_peak_memory_stats()
            training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
            training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
            if rank == 0:
                print(' '.join(fields))

            # Check for abort.
            if (not done) and (abort_fn is not None) and abort_fn():
                done = True
                if rank == 0:
                    print()
                    print('Aborting...')

            # Save image snapshot.
            if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0 or cur_tick == 1):
                with torch.no_grad():
                    # There is syncbn in id encoder when trained with multiple gpus, so if not be set to evaluation mode this process will be blocked.
                    id_encoder.eval()
                    id_feature = id_encoder(images)
                    grid_z = (id_feature).split(1)
                    out = [G(z=z, c=c, noise_mode='const') for z, c in zip(grid_z, grid_c)]
                    images_sr = torch.cat([o['image'].cpu() for o in out])
                    images_raw = torch.cat([o['image_raw'].cpu() for o in out])
                    images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
                    ssim_val = ssim(images_sr.to(device) * 0.5 + 0.5, images * 0.5 + 0.5, data_range=1.0, size_average=False) # return (N,)
                    stats_tfevents.add_scalar('valid/ssim', ssim_val.mean(), global_step=int(cur_nimg / 1e3), walltime=time.time() - start_time)
                    ssim_val_raw = ssim(images_raw.to(device) * 0.5 + 0.5, real_raw_images * 0.5 + 0.5, data_range=1.0, size_average=False) # return (N,)
                    stats_tfevents.add_scalar('valid/ssim_val_raw', ssim_val_raw.mean(), global_step=int(cur_nimg / 1e3), walltime=time.time() - start_time)
                    save_image_grid(images_sr.numpy(), os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
                    save_image_grid(images_raw.numpy(), os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw.png'), drange=[-1,1], grid_size=grid_size)
                    save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

                    id_feature = id_encoder(train_images)
                    grid_z = (id_feature).split(1)
                    out = [G(z=z, c=c, noise_mode='const') for z, c in zip(grid_z, train_grid_c)]
                    images_sr = torch.cat([o['image'].cpu() for o in out])
                    images_raw = torch.cat([o['image_raw'].cpu() for o in out])
                    images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
                    save_image_grid(images_sr.numpy(), os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_train.png'), drange=[-1,1], grid_size=grid_size)
                    save_image_grid(images_raw.numpy(), os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw_train.png'), drange=[-1,1], grid_size=grid_size)
                    save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth_train.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)
                    id_encoder.train()

            # Save network snapshot.
            snapshot_pkl = None
            snapshot_data = None
            name_list = []
            if gan_depth:
                name_list += [('D', D)]
            if train_en:
                name_list += [('E', id_encoder)]
                if not train_gen and G.z_dim != 512:
                    name_list += [('G_ema', G)]
            if train_gen:
                name_list += [('G_ema', G)]

            if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
                snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
                for name, module in name_list:
                    if module is not None:
                        if num_gpus > 1:
                            misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    snapshot_data[name] = module
                    del module # conserve memory
                    snapshot_pkl = os.path.join(run_dir, f'network-{name}-best.pkl')
                    if rank == 0 and ssim_val.mean().item() >= max_ssim:
                        max_ssim = ssim_val.mean().item()
                        with open(snapshot_pkl, 'wb') as f:
                            pickle.dump(snapshot_data, f)
                    snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))

            if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0 or cur_tick == 1):
                snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
                for name, module in name_list:
                    if module is not None:
                        if num_gpus > 1:
                            misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    snapshot_data[name] = module
                    del module # conserve memory
                    snapshot_pkl = os.path.join(run_dir, f'network-{name}-latest.pkl')
                    if rank == 0:
                        with open(snapshot_pkl, 'wb') as f:
                            pickle.dump(snapshot_data, f)
                    snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))

            if (network_snapshot_ticks is not None) and cur_tick % (500) == 0.0:
                snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
                for name, module in name_list:
                    if module is not None:
                        if num_gpus > 1:
                            misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    snapshot_data[name] = module
                    del module # conserve memory
                    snapshot_pkl = os.path.join(run_dir, f'network-{name}-{cur_tick}.pkl')
                    if rank == 0:
                        with open(snapshot_pkl, 'wb') as f:
                            pickle.dump(snapshot_data, f)
                    snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))

        except OSError:
            print("disk full, skip saving network!")
            pass

        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
