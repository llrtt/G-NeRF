# Official PyTorch implementation of "G-NeRF: Geometry-enhanced Novel View Synthesis from Single-View Images" (CVPR 2024)
Zixiong Huang*, Qi Chen*, Libo Sun, Yifan Yang, Naizhou Wang, Mingkui Tan, Qi Wu

![architecture](figure/overallarchitecture.png)
### [Project Page](https://llrtt.github.io/G-NeRF-Demo/)| [arXiv Paper](https://arxiv.org/abs/2310.08528)

## Requirements

* CUDA toolkit 11.3 or later.  (Why is a separate CUDA toolkit installation required?  We use the custom CUDA extensions from the StyleGAN3 repo. Please see [Troubleshooting](https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary)).
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


## Training

For training synthetic scenes such as `bouncingballs`, run

```
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py 
```

For training dynerf scenes such as `cut_roasted_beef`, run
```python
# First, extract the frames of each video.
python scripts/preprocess_dynerf.py --datadir data/dynerf/cut_roasted_beef
# Second, generate point clouds from input data.
bash colmap.sh data/dynerf/cut_roasted_beef llff
# Third, downsample the point clouds generated in the second step.
python scripts/downsample_point.py data/dynerf/cut_roasted_beef/colmap/dense/workspace/fused.ply data/dynerf/cut_roasted_beef/points3D_downsample2.ply
# Finally, train.
python train.py -s data/dynerf/cut_roasted_beef --port 6017 --expname "dynerf/cut_roasted_beef" --configs arguments/dynerf/cut_roasted_beef.py 
```
For training hypernerf scenes such as `virg/broom`, run
```python
# First, computing dense point clouds by COLMAP
bash colmap.sh data/hypernerf/virg/broom2 hypernerf
# Second, downsample the point clouds generated in the first step.
python scripts/downsample_point.py data/hypernerf/virg/broom2/colmap/dense/workspace/fused.ply data/hypernerf/virg/broom2/points3D_downsample2.ply
# Finally, train.
python train.py -s  data/hypernerf/virg/broom2/ --port 6017 --expname "hypernerf/broom2" --configs arguments/hypernerf/broom2.py 
```

For your custom datasets, install nerfstudio and follow their colmap pipeline.

```python
pip install nerfstudio
# computing camera poses by colmap pipeline
ns-process-data images --data data/your-data --output-dir data/your-ns-data
cp -r data/your-ns-data/images data/your-ns-data/colmap/images
python train.py -s data/your-ns-data/colmap --port 6017 --expname "custom" --configs arguments/hypernerf/default.py 
```
You can customize your training config through the config files.

## Evaluation

You can just run the following script to evaluate the model.

```
python metrics.py --model_path "output/dnerf/bouncingballs/" 
```



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