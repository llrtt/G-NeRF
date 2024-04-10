export CUDA_HOME=/mnt/cephfs/smil/cuda/cuda-11.1
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=/mnt/cephfs/home/llrt/anaconda3/envs/talkingnerf/lib/python3.10/site-packages/nvidia/cublas/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
CUDA_VISIBLE_DEVICES=1 python ./talkingnerf/fid_evaluation.py \
--network experiments/20230319_gpu022_train_gen_en_w_5120_z_5120_1000_face/00000-ffhq-eg3d-gpus5-batch5-gamma5/network-G_ema-latest.pkl \
--id_encoder experiments/20230319_gpu022_train_gen_en_w_5120_z_5120_1000_face/00000-ffhq-eg3d-gpus5-batch5-gamma5/network-E-latest.pkl \