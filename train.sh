export CUDA_HOME=/mnt/cephfs/smil/cuda/cuda-11.1
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=/mnt/cephfs/home/llrt/anaconda3/envs/talkingnerf/lib/python3.10/site-packages/nvidia/cublas/lib/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python talkingnerf/train.py \
--outdir=./experiments/20230310_gpu019_train_gen_1024_fix_en_ssim \
--cfg=ffhq \
--data=/mnt/cephfs/dataset/Face/EG3D_GEN_W_0.5/ \
--gpus=8 \
--batch=16 \
--gamma=5 \
--gen_pose_cond=False \
--metrics='' \
--workers=2 \
--aug=noaug \
--mbstd-group 1 \
--train_en False \
--train_gen True \
--resume_gen talkingnerf/ffhq512-128.pkl \
--resume_en experiments/20230310_gpu018_train_encoder_512_fix_gen_ssim/00010-ffhq-eg3d-gpus4-batch8-gamma5/network-E-000300.pkl \
--z_dim 512

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python talkingnerf/train.py \
--outdir=./experiments/20230312_gpu019_train_gen_en_w_1024_z_1024 \
--cfg=ffhq \
--data=/mnt/cephfs/dataset/Face/EG3D_GEN_W_0.5/ \
--gpus=8 \
--batch=8 \
--gamma=5 \
--gen_pose_cond=False \
--metrics='' \
--workers=2 \
--aug=noaug \
--mbstd-group 1 \
--train_en True \
--train_gen True \
--resume_gen talkingnerf/ffhq512-128.pkl \
--z_dim 1024 \
--w_dim 1024

CUDA_VISIBLE_DEVICES=2 python talkingnerf/train.py \
--outdir=./experiments/20230416_gpu026_tmp \
--cfg=ffhq \
--dataset_name=afhqv2 \
--data=/mnt/cephfs/dataset/Face/EG3D_GEN_CONDITION/ \
--gpus=1 \
--batch=1 \
--gamma=5 \
--gen_pose_cond=False \
--metrics='' \
--workers=2 \
--aug=noaug \
--mbstd-group 1 \
--train_en True \
--train_gen True \
--resume_gen afhqcats512-128.pkl \
--z_dim 512 \
--w_dim 512 \
--sr-module training.superresolution.SuperresolutionHybrid8X \
--cycle_loss False

CUDA_VISIBLE_DEVICES=6,7 python talkingnerf/train.py \
--outdir=experiments/ablation_studies/20230611_pzl009_512_5120_cats_128 \
--cfg=afhqv2 \
--dataset_name=afhqv2 \
--data=/home/huangzixiong/dataset/classnerf/AFHQV2_GEN_DATASET/AFHQV2_GEN_W_0.5/ \
--gpus=2 \
--batch=24 \
--gamma=5 \
--gen_pose_cond=False \
--metrics='' \
--workers=2 \
--aug=noaug \
--mbstd-group 1 \
--train_en True \
--train_gen True \
--z_dim 5120 \
--w_dim 5120 \
--snap 10 \
--neural_rendering_resolution_initial=128 \
--gan_depth True \
--kimg 2000

CUDA_VISIBLE_DEVICES=2,3 python talkingnerf/train.py \
--outdir=experiments/202300604_pzl009_chairs_128_5120_final \
--cfg=ffhq \
--dataset_name=ffhq \
--data=/home/huangzixiong/dataset/classnerf/ShapeNet_GEN_DATASET/SRN_CHAIRS_0.8 \
--gpus=2 \
--batch=24 \
--gamma=5 \
--gen_pose_cond=False \
--metrics='' \
--workers=2 \
--aug=noaug \
--mbstd-group 1 \
--train_en True \
--train_gen True \
--z_dim 5120 \
--w_dim 5120 \
--snap 10 \
--neural_rendering_resolution_initial=64 \
--gan_depth True

# update model
CUDA_VISIBLE_DEVICES=5 python talkingnerf/train.py \
--outdir=experiments/20230522_pzl009_tmp \
--cfg=ffhq \
--dataset_name=ffhq \
--data=/home/huangzixiong/dataset/classnerf/EG3D_GEN_DATASET/EG3D_GEN_W_0.5/ \
--gpus=1 \
--batch=2 \
--gamma=5 \
--gen_pose_cond=False \
--metrics='' \
--workers=2 \
--aug=noaug \
--mbstd-group 1 \
--train_en True \
--train_gen True \
--z_dim 5120 \
--w_dim 5120 \
--snap 10 \
--update_model True \
--neural_rendering_resolution_initial=64 \
--gan_depth True \
--resume_en experiments/ablation_studies/20230510_pzl009_512_5120_final/00003-ffhq-ffhq-gpus2-batch24-gamma5/network-E-latest.pkl \
--resume_gen experiments/ablation_studies/20230510_pzl009_512_5120_final/00003-ffhq-ffhq-gpus2-batch24-gamma5/network-G_ema-latest.pkl

CUDA_VISIBLE_DEVICES=6,7 python talkingnerf/train.py \
--outdir=experiments/ablation_studies/20230613_pzl009_512_7168_ffhq \
--cfg=ffhq \
--dataset_name=ffhq \
--data=/home/huangzixiong/dataset/classnerf/EG3D_GEN_DATASET/EG3D_GEN_W_0.5/ \
--gpus=2 \
--batch=24 \
--gamma=5 \
--gen_pose_cond=False \
--metrics='' \
--workers=2 \
--aug=noaug \
--mbstd-group 1 \
--train_en True \
--train_gen True \
--z_dim 7168 \
--w_dim 7168 \
--snap 10 \
--neural_rendering_resolution_initial=64 \
--gan_depth True \
--kimg 4000