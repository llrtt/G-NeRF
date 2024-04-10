outdir="/data_14t/data_hzx/zs3nerf/ID_eval/FFHQ"
CUDA_VISIBLE_DEVICES=4 python talkingnerf/ssim_evaluation.py \
  --network experiments/FFHQ/network-G_ema-3971.pkl \
  --id_encoder experiments/FFHQ/network-E-3971.pkl \
  --outdir ${outdir} \
  --dataset ffhq \
  --isRand True \
  --only_depth False \
  --res 64 \
  --dataset_dir '/data_14t/data_hzx/zs3nerf/FFHQ_in_the_wlid/cropped_image/' \
  --label_path '/data_14t/data_hzx/zs3nerf/FFHQ_in_the_wlid/label/labels.json'