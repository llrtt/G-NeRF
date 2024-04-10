import os
import json
import glob
import numpy as np
from tqdm import tqdm

def aggregate_pose_label(root_dir:str):
    all_label_paths = glob.glob(os.path.join(root_dir, "*/*.json"))
    all_label_paths.sort()
    all_label = {}
    for label_path in tqdm(all_label_paths):
        with open(label_path, 'r') as file:
            label_dict = json.load(file)
            for k, v in label_dict.items():
                all_label[k] = v
    with open(os.path.join(root_dir, "pose_labels.json"), 'w') as file:
        json.dump(all_label, file)

def aggregate_depth_images(root_dir:str):
    all_label_paths = glob.glob(os.path.join(root_dir, "*/*.npy"))
    all_label_paths.sort()
    all_images = {}
    for label_path in tqdm(all_label_paths):
        base_name = os.path.basename(label_path).replace('.npy', '')
        all_images[base_name] = np.load(label_path)
    np.save(os.path.join(root_dir, "depth_images.npy"), all_images)

if __name__ == '__main__':
    aggregate_pose_label('/mnt/cephfs/dataset/Face/EG3D_FFHQ_label')
