# outdir="evaluation_results/quanlitative_results/afhqv2"
# CUDA_VISIBLE_DEVICES=0 python talkingnerf/ssim_evaluation.py \
# --network experiments/20230510_pzl009_cats_final/00001-afhqv2-afhqv2-gpus2-batch48-gamma5/network-G_ema-latest.pkl \
# --id_encoder experiments/20230510_pzl009_cats_final/00001-afhqv2-afhqv2-gpus2-batch48-gamma5/network-E-latest.pkl \
# --outdir ${outdir} \
# --dataset afhqv2 \
# --isRand False \
# --dataset_dir /home/huangzixiong/dataset/classnerf/afhq_v2/train/cat \
# --label_path /home/huangzixiong/dataset/classnerf/afhq_v2/train/label/labels.json \

# python -m pytorch_fid "${outdir}/fake/64" "${outdir}/real/64"
# python -m pytorch_fid "${outdir}/fake/512" "${outdir}/real/512"
# fidelity --gpu 0 --kid --input1 "${outdir}/real/64" --input2 "${outdir}/fake/64"
# fidelity --gpu 0 --kid --input1 "${outdir}/real/512" --input2 "${outdir}/fake/512"

outdir="/data/baowenjie/20231115_backup/evaluation_results/c_eg3d/random"
CUDA_VISIBLE_DEVICES=0 python talkingnerf/ssim_evaluation.py \
  --network "/data/baowenjie/20231115_backup/TalkingNeRF/experiments/20231031_CVTE_A800_C_EG3D_Random_0.6_No_SyncBN/00000-ffhq-ffhq-gpus2-batch24-gamma5/network-G_ema-latest.pkl" \
  --id_encoder "/data/baowenjie/20231115_backup/TalkingNeRF/experiments/20231031_CVTE_A800_C_EG3D_Random_0.6_No_SyncBN/00000-ffhq-ffhq-gpus2-batch24-gamma5/network-E-latest.pkl" \
  --outdir ${outdir} \
  --dataset ffhq \
  --isRand True \
  --only_depth False \
  --side_face False  \
  --res 64 \
  --dataset_dir '/data_local/dataset/FFHQ_in_the_wlid/cropped_image' \
  --label_path '/data_local/dataset/FFHQ_in_the_wlid/label/labels.json'

python -m pytorch_fid "${outdir}/fake/64" "${outdir}/real/64"
python -m pytorch_fid "${outdir}/fake/512" "${outdir}/real/512"

fidelity --gpu 0 --kid --input1 "${outdir}/real/64" --input2 "${outdir}/fake/64"
fidelity --gpu 0 --kid --input1 "${outdir}/real/512" --input2 "${outdir}/fake/512"

# outdir="evaluation_results/ablation_studies/celeba/no_depth_extreme"
# CUDA_VISIBLE_DEVICES=3 python talkingnerf/ssim_evaluation.py \
# --network experiments/ablation_studies/20230510_pzl009_512_5120_no_depth/00000-ffhq-ffhq-gpus2-batch24-gamma5/network-G_ema-latest.pkl \
# --id_encoder experiments/ablation_studies/20230510_pzl009_512_5120_no_depth/00000-ffhq-ffhq-gpus2-batch24-gamma5/network-E-latest.pkl \
# --outdir ${outdir} \
# --dataset celeba \
# --isRand True \
# --dataset_dir /home/huangzixiong/dataset/classnerf/CelebAMask-HQ/cropped_700_images_test \
# --label_path /home/huangzixiong/dataset/classnerf/CelebAMask-HQ/label/labels.json \

# python -m pytorch_fid "${outdir}/fake/64" "${outdir}/real/64"
# python -m pytorch_fid "${outdir}/fake/512" "${outdir}/real/512"

# fidelity --gpu 0 --kid --input1 "${outdir}/real/64" --input2 "${outdir}/fake/64"
# fidelity --gpu 0 --kid --input1 "${outdir}/real/512" --input2 "${outdir}/fake/512"

# outdir="evaluation_results/ablation_studies/celeba/w_5120"
# CUDA_VISIBLE_DEVICES=3 python talkingnerf/ssim_evaluation.py \
# --network experiments/ablation_studies/20230517_pzl009_512_5120_128/00000-ffhq-ffhq-gpus2-batch24-gamma5/network-G_ema-latest.pkl \
# --id_encoder experiments/ablation_studies/20230517_pzl009_512_5120_128/00000-ffhq-ffhq-gpus2-batch24-gamma5/network-E-latest.pkl \
# --outdir ${outdir} \
# --dataset celeba \
# --isRand False \
# --only_depth False \
# --res 128 \
# --dataset_dir '/home/huangzixiong/dataset/classnerf/CelebAMask-HQ/cropped_700_images_test/' \
# --label_path '/home/huangzixiong/dataset/classnerf/CelebAMask-HQ/label/labels.json'

# python -m pytorch_fid "${outdir}/fake/128" "${outdir}/real/128"
# python -m pytorch_fid "${outdir}/fake/512" "${outdir}/real/512"
# fidelity --gpu 0 --kid --input1 "${outdir}/real/128" --input2 "${outdir}/fake/128"
# fidelity --gpu 0 --kid --input1 "${outdir}/real/512" --input2 "${outdir}/fake/512"