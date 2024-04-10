outdir="/data/baowenjie/20231115_backup/evaluation_results/pti/reconstruction"
CUDA_VISIBLE_DEVICES=0 python talkingnerf/pti/eval.py \
  --network "/data/baowenjie/20231115_backup/experiments/Ablation/only_multi-view/network-G_ema-best.pkl" \
  --id_encoder "/data/baowenjie/20231115_backup/experiments/Ablation/only_multi-view/network-E-best.pkl" \
  --outdir ${outdir} \
  --dataset ffhq \
  --isRand False \
  --only_depth False \
  --side_face False  \
  --res 64 \
  --dataset_dir '/data_local/dataset/FFHQ_in_the_wlid/pti_eval' \
  --label_path '/data_local/dataset/FFHQ_in_the_wlid/label/labels.json'

python -m pytorch_fid "${outdir}/fake/64" "${outdir}/real/64"
python -m pytorch_fid "${outdir}/fake/512" "${outdir}/real/512"

fidelity --gpu 0 --kid --input1 "${outdir}/real/64" --input2 "${outdir}/fake/64"
fidelity --gpu 0 --kid --input1 "${outdir}/real/512" --input2 "${outdir}/fake/512"