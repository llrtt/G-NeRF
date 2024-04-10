
CUDA_VISIBLE_DEVICES=1 python ./g_nerf/gen_talking_videos.py \
--network checkpoints/G-NeRF/network-G_ema-final.pkl \
--id_encoder checkpoints/G-NeRF/network-E-final.pkl \
--id_image samples/66667.jpg \
--outdir results \
--video_out_path results \