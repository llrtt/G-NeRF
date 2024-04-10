# CUDA_VISIBLE_DEVICES=6 python ./talkingnerf/gen_talking_videos.py \
# --network experiments/20230510_pzl009_cats_final/00003-afhqv2-afhqv2-gpus2-batch48-gamma5/network-G_ema-latest.pkl \
# --id_encoder experiments/20230510_pzl009_cats_final/00003-afhqv2-afhqv2-gpus2-batch48-gamma5/network-E-latest.pkl \
# --id_image /home/huangzixiong/dataset/classnerf/afhq_v2/train/cat/pixabay_cat_004833.png \
# --outdir evaluation_results/qualitvative_results/afhqv2 \
# --video_out_path video_results/srnchairs/sup \
# --gen_shapes False \
# --label_path /home/huangzixiong/dataset/classnerf/CelebAMask-HQ/label/labels.json

# --prepared samples \
# --id_image /home/huangzixiong/dataset/classnerf/FFHQ_in_the_wlid/cropped_image/66551.jpg \
# --id_image  /mnt/cephfs/dataset/Face/EG3D_GEN_CONDITION/000000/000000_f.jpg \
# --id_image /mnt/cephfs/dataset/Face/afhq_v2/train/cat/pixabay_cat_001624.png \
# /mnt/cephfs/dataset/Face/EG3D_GEN_CONDITION_RANDOM_POSE_LARGE/000041/000041_f.jpg

id_image="/data_14t/data_hzx/zs3nerf/FFHQ_in_the_wlid/cropped_image/66036.jpg"
CUDA_VISIBLE_DEVICES=7 python ./talkingnerf/gen_talking_videos.py \
--network /data_14t/data_hzx/zs3nerf/experiments/FFHQ/network-G_ema-3971.pkl \
--id_encoder /data_14t/data_hzx/zs3nerf/experiments/FFHQ/network-E-3971.pkl \
--id_image ${id_image} \
--outdir /data_14t/data_hzx/zs3nerf/evaluation_results/ablation_studies/visualization/0_5 \
--video_out_path video_results/ablation_study/no_depth/sup \
--label_path /data_14t/data_hzx/zs3nerf/FFHQ_in_the_wlid/label/labels.json \
--gen_shapes True \

# CUDA_VISIBLE_DEVICES=3 python ./talkingnerf/gen_talking_videos.py \
# --network experiments/ablation_studies/20230510_pzl009_512_5120_final/00000-ffhq-ffhq-gpus2-batch24-gamma5/network-G_ema-3971.pkl \
# --id_encoder experiments/ablation_studies/20230510_pzl009_512_5120_final/00000-ffhq-ffhq-gpus2-batch24-gamma5/network-E-3971.pkl \
# --id_image ${id_image} \
# --outdir evaluation_results/qualitvative_results/ffhq \
# --video_out_path video_results/ablation_study/w_5120/sup \
# --label_path /home/huangzixiong/dataset/classnerf/CelebAMask-HQ/label/labels.json \
# --gen_shapes False \

# CUDA_VISIBLE_DEVICES=3 python ./talkingnerf/gen_talking_videos.py \
# --network experiments/ablation_studies/20230507_pzl009_512_5120_only_multiview/00002-ffhq-ffhq-gpus2-batch24-gamma5/network-G_ema-latest.pkl \
# --id_encoder experiments/ablation_studies/20230507_pzl009_512_5120_only_multiview/00002-ffhq-ffhq-gpus2-batch24-gamma5/network-E-latest.pkl \
# --id_image ${id_image} \
# --outdir evaluation_results/qualitvative_results/ffhq \
# --video_out_path video_results/ablation_study/only_multiview/sup \
# --label_path /home/huangzixiong/dataset/classnerf/CelebAMask-HQ/label/labels.json \
# --gen_shapes False \

# cp ${id_image} video_results/ablation_study/

# CUDA_VISIBLE_DEVICES=0 python ./talkingnerf/gen_talking_videos.py \
# --network experiments/202300604_pzl009_cars_128_5120_final/00000-shapenet-shapenet-gpus2-batch24-gamma5/network-G_ema-best.pkl \
# --id_encoder experiments/202300604_pzl009_cars_128_5120_final/00000-shapenet-shapenet-gpus2-batch24-gamma5/network-E-best.pkl \
# --id_image /home/huangzixiong/dataset/classnerf/srn/cars_test/460f7950d804d4564d7ac55d461d1984/rgb/000047.png \
# --outdir evaluation_results/qualitvative_results/srn_chairs/91c12a0bdf98f5d220f29d4da2b76f7a/ \
# --video_out_path video_results/srncars/91c12a0bdf98f5d220f29d4da2b76f7a/ \
# --label_path /home/huangzixiong/dataset/classnerf/srn_chairs/chairs_test/label/labels.json \
# --gen_shapes False \
# --dataset shapenet

# cp /home/huangzixiong/dataset/classnerf/srn/cars_test/91c12a0bdf98f5d220f29d4da2b76f7a/rgb/000060.png  video_results/srncars/91c12a0bdf98f5d220f29d4da2b76f7a/000060.png