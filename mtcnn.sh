# conda activate preprocess_113
export CUDA_HOME=/mnt/cephfs/smil/cuda/cuda-11.1
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
python data_preprocessing/celebamask_hq/mtcnn.py