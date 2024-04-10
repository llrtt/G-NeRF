export CUDA_HOME=/mnt/cephfs/smil/cuda/cuda-11.1
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=/mnt/cephfs/home/llrt/anaconda3/envs/talkingnerf/lib/python3.10/site-packages/nvidia/cublas/lib/

CUDA_VISIBLE_DEVICES=3 python talkingnerf/train_generator/train.py \
--network talkingnerf/ffhq512-128.pkl \
--encoder_path experiments/20230219_gpu020_train_encoder/00005-ffhq-eg3d-gpus2-batch8-gamma1/network-snapshot-000720.pkl \
--dataset_path /mnt/cephfs/dataset/Face/EG3D_GEN --batch_size 4
