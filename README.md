# Official PyTorch implementation of "G-NeRF: Geometry-enhanced Novel View Synthesis from Single-View Images" (CVPR 2024)
Zixiong Huang*, Qi Chen*, Libo Sun, Yifan Yang, Naizhou Wang, Mingkui Tan, Qi Wu

![architecture](figure/overallarchitecture.png)
### [Project Page](https://llrtt.github.io/G-NeRF-Demo/)| [arXiv Paper](https://arxiv.org/abs/2404.07474)

## Updates / TODO List

- ✅ [2024/04/15] Update inference code.

- 🔲 We will release the data generation code and training code later.

## Requirements

* CUDA toolkit 11.3 or later.
* Python libraries: see [environment.yml](./eg3d/environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `cd g_nerf`
  - `conda env create -f environment.yml`
  - `conda activate gnerf`

In our environment, we use pytorch=1.13.1+cu116.

## Inference

Download our pre-trained checkpoint from [huggingface](https://huggingface.co/llrt/G-NeRF) and put them into checkpoints dir.

```.bash
# Generate videos using pre-trained model

python ./g_nerf/gen_videos.py \
--network checkpoints/G-NeRF/network-G_ema-final.pkl \
--id_encoder checkpoints/G-NeRF/network-E-final.pkl \
--id_image samples/66667.jpg \
--outdir results \
--video_out_path results
```

## Synthetic Date Generation
Coming Soon

## Training
Coming Soon

## Citation

```
@inproceedings{huang2024g,
  author = {Zixiong, Huang and Qi, Chen and Libo, Sun and Yifan, Yang and Naizhou, Wang and Mingkui, Tan and Qi, Wu},
  title = {G-NeRF: Geometry-enhanced Novel View Synthesis from Single-View Images},
  booktitle = {IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year = {2024}
}
```

## Acknowledgement

 Our code is modified from [EG3D](https://github.com/NVlabs/eg3d/tree/main). Thanks for their awesome work!