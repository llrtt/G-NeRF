# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""
import os
import cv2
cv2.setNumThreads(1)
import sys
import re
sys.path.append("talkingnerf")
import glob
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from camera_utils import LookAtPoseSampler
try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

class MeadDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = 512,  # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        self._video_fnames = []
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = glob.glob(os.path.join(self._path, r"*/video/*/*/level_2/*crop.mp4"))
            self._all_fnames.sort()
            for p in self._all_fnames:
                feature_name = p.replace(re.findall(r'.*/(video/.*?)/', p)[0], 'audio').replace('_crop.mp4', '.npy')
                if os.path.isfile(feature_name):
                     self._video_fnames.append(p)
        else:
            raise IOError('Path must point to a directory')

        if len(self._video_fnames) == 0:
            raise IOError('No image files found in the specified path')

        self.name_dict = {}
        self.name2label = {}
        label_counter = 0
        for i, p in enumerate(self._video_fnames):
            id_label = re.findall(r'/.*/(.*?)/video/', p)[0]
            if id_label in self.name_dict.keys():
                self.name_dict[id_label].append(i)
            else:
                self.name_dict[id_label] = [i]
                self.name2label[id_label] = label_counter
                label_counter+=1

        name = "MEAD"
        raw_shape = [len(self._video_fnames)] + [3] + [512] * 2
        self._raw_labels_std = torch.zeros((25))
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def frame2audio_indexs(self, frame_inds):
        start_frame_ind = frame_inds - 5 // 2

        start_audio_inds = start_frame_ind * 4
        return start_audio_inds

    def load_spectrogram(self, audio_ind):
        mel_shape = self.audio_spect.shape
        if (audio_ind + 20) <= mel_shape[0] and audio_ind >= 0:
            spectrogram = np.array(self.audio_spect[audio_ind:audio_ind + 20, :]).astype('float32')
        else:
            print('(audio_ind {} + opt.num_audio_bins {}) > mel_shape[0] {} '.format(audio_ind, 20,
                                                                                     mel_shape[0]))
            if audio_ind > 0:
                spectrogram = np.array(self.audio_spect[audio_ind:audio_ind + 20, :]).astype('float32')
            else:
                spectrogram = np.zeros((20, mel_shape[1])).astype(np.float16).astype(np.float32)

        spectrogram = torch.from_numpy(spectrogram)
        spectrogram = spectrogram.unsqueeze(0)

        spectrogram = spectrogram.transpose(-2, -1)
        return spectrogram

    def get_identity(self, index):
        self.id_label = re.findall(r'/.*/(.*?)/video/', self._video_fnames[index])[0]
        r_index = np.random.randint(0, len(self.name_dict[self.id_label]) - 1)
        r_video = self._video_fnames[self.name_dict[self.id_label][r_index]]
        video = cv2.VideoCapture(r_video)
        identity = video.read()[1]
        identity = cv2.cvtColor(identity, cv2.COLOR_BGR2RGB)
        return identity.transpose(2, 0, 1)

    def _load_all(self, idx):
        try:
            video_fname = self._video_fnames[self._raw_idx[idx]]
            fname = video_fname.replace('video', 'label').replace('crop.mp4', 'dataset.json')
            feature_name = video_fname.replace(re.findall(r'.*/(video/.*?)/', video_fname)[0], 'audio').replace('_crop.mp4', '.npy')
            if not os.path.isfile(video_fname) or not os.path.isfile(fname) or not os.path.isfile(feature_name):
                print(f"File may not exist: {video_fname}")
                raise Exception("\n")
            
            cap = cv2.VideoCapture(video_fname)
            frame_num = int(cap.get(7))

            with open(fname, 'r') as f:
                labels = json.load(f)['labels']
            label_num = len(labels)

            self.audio_spect  = np.load(feature_name)
            spect_num = len(self.audio_spect) // 4 - 2

            max_num = frame_num if frame_num < spect_num else spect_num
            max_num = label_num if label_num < max_num else max_num

            self.target_frame_inds = np.arange(2, max_num)
            self.audio_inds = self.frame2audio_indexs(self.target_frame_inds)

            # Choose a random index for img and spectrogram
            r_index = np.random.randint(0, len(self.target_frame_inds)-1)
            img_index = self.target_frame_inds[r_index]
            mel_index = self.audio_inds[r_index]

            # load spectrograms
            spectrograms = self.load_spectrogram(mel_index)

            # load_raw_image
            cap.set(cv2.CAP_PROP_POS_FRAMES, img_index)
            ret, image = cap.read()
            if not ret:
                print(f"img indx {img_index} is out of range.(max: {frame_num})")
                raise Exception("\n")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose(2, 0, 1) # HWC => CHW

            id_index = np.random.randint(0, frame_num-1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, id_index)
            ret, id_image = cap.read()
            id_image = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)
            id_image = id_image.transpose(2, 0, 1) # HWC => CHW
            # # load_raw_labels
            labels = dict(labels)
            label = np.array(labels[str(img_index)])
            label = label.astype(np.float32)

            return image, id_image, label, spectrograms
        except Exception:
            print(f"Errupted data: {video_fname}")
            return None, None, None, None

    @property
    def has_labels(self):
        return True

    @property
    def label_shape(self):
        return [25]

    def get_label(self, idx):
        video_fname = self._video_fnames[self._raw_idx[idx]]
        fname = video_fname.replace('video', 'label').replace('crop.mp4', 'dataset.json')
        with open(fname, 'r') as f:
            labels = json.load(f)['labels']
        label_num = len(labels)
        while(label_num <= 0):
            return self.get_label(0)
        r_idx= np.random.randint(label_num-1)
        labels = dict(labels)
        while(not str(r_idx) in labels.keys()):
            r_idx= np.random.randint(label_num-1)
        label = np.array(labels[str(r_idx)])
        label = label.astype(np.float32)
        return label.copy()

    def __getitem__(self, idx):
        image, id_image, label, spectrograms = self._load_all(idx)
        while(image is None):
            idx = np.random.randint(0, len(self._video_fnames))
            image, id_image, label, spectrograms = self._load_all(idx)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image, label, spectrograms, id_image

class MixedDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = 512,  # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        self._video_fnames = []
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_mead_fnames = glob.glob(os.path.join(self._path, r"*/video/*/*/level_2/*crop.mp4"))
            self._all_mead_fnames.sort()
            for p in self._all_mead_fnames:
                feature_name = p.replace(re.findall(r'.*/(video/.*?)/', p)[0], 'audio').replace('_crop.mp4', '.npy')
                if os.path.isfile(feature_name):
                     self._video_fnames.append(p)
        else:
            raise IOError('Path must point to a directory')

        self._ffhq_fnames = []
        if os.path.isdir("/mnt/cephfs/dataset/Face/FFHQ"):
            self._type = 'dir'
            self._all_ffhq_fnames = glob.glob(os.path.join("/mnt/cephfs/dataset/Face/FFHQ", r"images_600/*.png"))
            self._all_ffhq_fnames.sort()
            for p in self._all_ffhq_fnames:
                label_name = p.replace("images_600", "normalized_labels").replace('.png', '.json')
                if os.path.isfile(label_name):
                     self._ffhq_fnames.append(p)
        else:
            raise IOError('Path must point to a directory')

        if len(self._video_fnames) == 0 or len(self._ffhq_fnames) == 0:
            raise IOError('No image files found in the specified path')

        self.name_dict = {}
        self.name2label = {}
        label_counter = 0
        for i, p in enumerate(self._video_fnames):
            id_label = re.findall(r'/.*/(.*?)/video/', p)[0]
            if id_label in self.name_dict.keys():
                self.name_dict[id_label].append(i)
            else:
                self.name_dict[id_label] = [i]
                self.name2label[id_label] = label_counter
                label_counter+=1

        name = "Mixed"
        raw_shape = [len(self._video_fnames)] + [3] + [512] * 2
        self._raw_labels_std = torch.zeros((25))
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def frame2audio_indexs(self, frame_inds):
        start_frame_ind = frame_inds - 5 // 2

        start_audio_inds = start_frame_ind * 4
        return start_audio_inds

    def load_spectrogram(self, audio_ind):
        mel_shape = self.audio_spect.shape
        if (audio_ind + 20) <= mel_shape[0] and audio_ind >= 0:
            spectrogram = np.array(self.audio_spect[audio_ind:audio_ind + 20, :]).astype('float32')
        else:
            print('(audio_ind {} + opt.num_audio_bins {}) > mel_shape[0] {} '.format(audio_ind, 20,
                                                                                     mel_shape[0]))
            if audio_ind > 0:
                spectrogram = np.array(self.audio_spect[audio_ind:audio_ind + 20, :]).astype('float32')
            else:
                spectrogram = np.zeros((20, mel_shape[1])).astype(np.float16).astype(np.float32)

        spectrogram = torch.from_numpy(spectrogram)
        spectrogram = spectrogram.unsqueeze(0)

        spectrogram = spectrogram.transpose(-2, -1)
        return spectrogram

    def get_identity(self, index):
        self.id_label = re.findall(r'/.*/(.*?)/video/', self._video_fnames[index])[0]
        r_index = np.random.randint(0, len(self.name_dict[self.id_label]) - 1)
        r_video = self._video_fnames[self.name_dict[self.id_label][r_index]]
        video = cv2.VideoCapture(r_video)
        identity = video.read()[1]
        identity = cv2.cvtColor(identity, cv2.COLOR_BGR2RGB)
        return identity.transpose(2, 0, 1)

    def _load_all_mead(self, idx):
        try:
            video_fname = self._video_fnames[self._raw_idx[idx]]
            fname = video_fname.replace('video', 'label').replace('crop.mp4', 'dataset.json')
            feature_name = video_fname.replace(re.findall(r'.*/(video/.*?)/', video_fname)[0], 'audio').replace('_crop.mp4', '.npy')
            if not os.path.isfile(video_fname) or not os.path.isfile(fname) or not os.path.isfile(feature_name):
                print(f"File may not exist: {video_fname}")
                raise Exception("\n")
            
            cap = cv2.VideoCapture(video_fname)
            frame_num = int(cap.get(7))

            with open(fname, 'r') as f:
                labels = json.load(f)['labels']
            label_num = len(labels)

            self.audio_spect  = np.load(feature_name)
            spect_num = len(self.audio_spect) // 4 - 2

            max_num = frame_num if frame_num < spect_num else spect_num
            max_num = label_num if label_num < max_num else max_num

            self.target_frame_inds = np.arange(2, max_num)
            self.audio_inds = self.frame2audio_indexs(self.target_frame_inds)

            # Choose a random index for img and spectrogram
            r_index = np.random.randint(0, len(self.target_frame_inds)-1)
            img_index = self.target_frame_inds[r_index]
            mel_index = self.audio_inds[r_index]

            # load spectrograms
            spectrograms = self.load_spectrogram(mel_index)

            # load_raw_image
            cap.set(cv2.CAP_PROP_POS_FRAMES, img_index)
            ret, image = cap.read()
            if not ret:
                print(f"img indx {img_index} is out of range.(max: {frame_num})")
                raise Exception("\n")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose(2, 0, 1) # HWC => CHW
            id_image = self.get_identity(self._raw_idx[idx])
            # # load_raw_labels
            labels = dict(labels)
            label = np.array(labels[str(img_index)])
            label = label.astype(np.float32)

            return image, id_image, label, spectrograms
        except Exception:
            print(f"Errupted data: {video_fname}")
            return None, None, None, None

    def _load_all_ffhq(self, idx):
        try:
            image_fname = self._ffhq_fnames[self._raw_idx[idx]]
            label_fname = image_fname.replace("images_600", "normalized_labels").replace('.png', '.json')
            if not os.path.isfile(image_fname) or not os.path.isfile(label_fname):
                print(f"File may not exist: {label_fname}")
                raise Exception("\n")
            
            image = cv2.imread(image_fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            id_image = image.copy()
            w, h = id_image.shape[:-1]
            image = image.transpose(2, 0, 1) # HWC => CHW

            # rand_angle = np.random.randint(-45, 45)
            # M_1 = cv2.getRotationMatrix2D((w/2, h/2), rand_angle, 1)
            # id_image = cv2.warpAffine(id_image, M_1, (w, h))
            id_image = id_image.transpose(2, 0, 1) # HWC => CHW

            with open(label_fname, 'r') as f:
                labels = json.load(f)['labels']

            # load_raw_labels
            labels = dict(labels)
            label = np.array(labels['image'])
            label = label.astype(np.float32)

            spectrograms = torch.zeros((1, 80, 20))

            return image, id_image, label, spectrograms
        except Exception:
            print(f"Errupted data: {image_fname}")
            return None, None, None, None

    @property
    def has_labels(self):
        return True

    @property
    def label_shape(self):
        return [25]

    def get_label(self, idx):
        video_fname = self._video_fnames[self._raw_idx[idx]]
        fname = video_fname.replace('video', 'label').replace('crop.mp4', 'dataset.json')
        with open(fname, 'r') as f:
            labels = json.load(f)['labels']
        label_num = len(labels)
        while(label_num <= 0):
            return self.get_label(0)
        r_idx= np.random.randint(label_num-1)
        labels = dict(labels)
        while(not str(r_idx) in labels.keys()):
            r_idx= np.random.randint(label_num-1)
        label = np.array(labels[str(r_idx)])
        label = label.astype(np.float32)
        return label.copy()

    def _load_all(self, idx):
        rand_c = np.random.randint(2)
        if False:
            return self._load_all_mead(idx)
        else:
            return self._load_all_ffhq(idx)

    def __getitem__(self, idx):
        image, id_image, label, spectrograms = self._load_all(idx)
        while(image is None):
            idx = np.random.randint(0, len(self._video_fnames))
            image, id_image, label, spectrograms = self._load_all(idx)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image, label, spectrograms, id_image

class CelebADataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = 512,  # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        self._video_fnames = []
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = glob.glob(os.path.join(self._path, r"*/*.jpg"))
            self._all_fnames.sort()
            for p in self._all_fnames:
                json_name = p.replace(re.findall(r'.*/(labeled_images/.*?)/', p)[0], 'normalized_labels').replace('jpg', 'json')
                if os.path.isfile(json_name):
                     self._video_fnames.append(p)
        else:
            raise IOError('Path must point to a directory')

        if len(self._video_fnames) == 0:
            raise IOError('No image files found in the specified path')

        self.name_dict = {}
        self.name2label = {}
        label_counter = 0
        for i, p in enumerate(self._video_fnames):
            id_label = re.findall(r'.*/labeled_images/(.*?)/', p)[0]
            if id_label in self.name_dict.keys():
                self.name_dict[id_label].append(i)
            else:
                self.name_dict[id_label] = [i]
                self.name2label[id_label] = label_counter
                label_counter+=1

        name = "CelebA"
        raw_shape = [len(self._video_fnames)] + [3] + [512] * 2
        self._raw_labels_std = torch.zeros((25))
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)


    def get_identity(self, index):
        self.id_label = re.findall(r'.*/labeled_images/(.*?)/', self._video_fnames[index])[0]
        if len(self.name_dict[self.id_label]) == 1:
            r_index = 0
        else:
            r_index = np.random.randint(0, len(self.name_dict[self.id_label]) - 1)
        r_video = self._video_fnames[self.name_dict[self.id_label][r_index]]
        video = cv2.VideoCapture(r_video)
        identity = video.read()[1]
        identity = cv2.cvtColor(identity, cv2.COLOR_BGR2RGB)
        return identity.transpose(2, 0, 1)

    def _load_all(self, idx):
        try:
            image_name = self._video_fnames[self._raw_idx[idx]]
            label_name = image_name.replace(re.findall(r'.*/(labeled_images/.*?)/', image_name)[0], 'normalized_labels').replace('jpg', 'json')
            if not os.path.isfile(image_name) or not os.path.isfile(label_name):
                print(f"File may not exist: {image_name}")
                raise Exception("\n")

            with open(label_name, 'r') as f:
                labels = json.load(f)['labels']

            # load spectrograms
            spectrograms = torch.zeros((1, 80, 20))

            # load_raw_image
            image = cv2.imread(image_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose(2, 0, 1) # HWC => CHW
            id_image = self.get_identity(self._raw_idx[idx])

            # # load_raw_labels
            labels = dict(labels)
            label = np.array(labels['image'])
            label = label.astype(np.float32)

            return image, id_image, label, spectrograms
        except Exception:
            print(f"Errupted data: {image_name}")
            return None, None, None, None

    @property
    def has_labels(self):
        return True

    @property
    def label_shape(self):
        return [25]

    def get_label(self, idx):
        image_name = self._video_fnames[self._raw_idx[idx]]
        label_name = image_name.replace(re.findall(r'.*/(labeled_images/.*?)/', image_name)[0], 'normalized_labels').replace('jpg', 'json')
        with open(label_name, 'r') as f:
            labels = json.load(f)['labels']
        label_num = len(labels)
        while(label_num <= 0):
            return self.get_label(0)
        labels = dict(labels)
        label = np.array(labels['image'])
        label = label.astype(np.float32)
        return label.copy()

    def __getitem__(self, idx):
        image, id_image, label, spectrograms = self._load_all(idx)
        while(image is None):
            idx = np.random.randint(0, len(self._video_fnames))
            image, id_image, label, spectrograms = self._load_all(idx)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image, label, spectrograms, id_image


class GenDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = 512,  # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        self._image_fnames = []
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._image_fnames = glob.glob(os.path.join(self._path, r"*/*.jpg"))
            self._image_fnames.sort()
            with open(os.path.join(self._path, 'pose_labels.json'), 'r') as file:
                self._pose_labels = json.load(file)
            self._depth_images = np.load(os.path.join(self._path, 'depth_images.npy'), allow_pickle=True)
        else:
            raise IOError('Path must point to a directory')

        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = "Gen"
        raw_shape = [len(self._image_fnames)] + [3] + [512] * 2
        self._raw_labels_std = torch.zeros((25))
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()
    
    @staticmethod
    def aug_image(image):
        # randomly scale
        random_s = np.random.randint(0, 15)
        image = cv2.resize(np.asarray(image[random_s:-(random_s+1), random_s:-(random_s+1), :]), (512, 512))
        # randomly rotate image
        random_state = np.random.randint(2, size=1)
        if(random_state):
            rotate_angle = np.random.randint(-15, 15)
            M = cv2.getRotationMatrix2D((256,256), rotate_angle, 1)
            image = cv2.warpAffine(image, M, (512,512))
        return image

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_all(self, idx):
        output = {}
        image_name = self._image_fnames[self._raw_idx[idx]]

        # load_raw_image
        condition_image = cv2.imread(image_name)
        condition_image = self.aug_image(condition_image)
        condition_image = cv2.cvtColor(condition_image, cv2.COLOR_BGR2RGB)
        condition_image = condition_image.transpose(2, 0, 1) # HWC => CHW
        loss_image = cv2.imread(image_name.replace('f.jpg', 's.jpg'))
        loss_image = cv2.cvtColor(loss_image, cv2.COLOR_BGR2RGB)
        loss_image = loss_image.transpose(2, 0, 1) # HWC => CHW

        output['condition_image'] = condition_image
        output['loss_image'     ] = loss_image
        output['condition_c'    ] = np.array(self._pose_labels[os.path.basename(image_name).replace('.jpg','.json')]).astype(np.float32)
        output['loss_c'         ] = np.array(self._pose_labels[os.path.basename(image_name).replace('.jpg','.json').replace('f', 's')]).astype(np.float32)
        output['c_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg',''))
        output['l_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg','').replace('f', 's'))

        return output

    @property
    def has_labels(self):
        return True

    @property
    def label_shape(self):
        return [25]

    def get_label(self, idx):
        image_name = self._image_fnames[self._raw_idx[idx]]
        label = np.array(self._pose_labels[os.path.basename(image_name).replace('.jpg','.json')])
        label = label.astype(np.float32)
        return label.copy()

    def __getitem__(self, idx):
        output = self._load_all(idx)
        assert isinstance(output['condition_image'], np.ndarray)
        assert list(output['condition_image'].shape) == self.image_shape
        assert output['condition_image'].dtype == np.uint8
        return output

class FFHQDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = 512,  # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        self._ffhq_fnames = []
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_ffhq_fnames = glob.glob(os.path.join(self._path, r"images_600/*.png"))
            self._all_ffhq_fnames.sort()
            for p in self._all_ffhq_fnames:
                label_name = p.replace("images_600", "normalized_labels").replace('.png', '.json')
                if os.path.isfile(label_name):
                     self._ffhq_fnames.append(p)
        else:
            raise IOError('Path must point to a directory')

        if len(self._ffhq_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = "FFHQ"
        raw_shape = [len(self._ffhq_fnames)] + [3] + [512] * 2
        self._raw_labels_std = torch.zeros((25))
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_all_ffhq(self, idx):
        try:
            image_fname = self._ffhq_fnames[self._raw_idx[idx]]
            label_fname = image_fname.replace("images_600", "normalized_labels").replace('.png', '.json')
            if not os.path.isfile(image_fname) or not os.path.isfile(label_fname):
                print(f"File may not exist: {label_fname}")
                raise Exception("\n")
            
            image = cv2.imread(image_fname)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.transpose(2, 0, 1) # HWC => CHW

            # rand_angle = np.random.randint(-45, 45)
            # M_1 = cv2.getRotationMatrix2D((w/2, h/2), rand_angle, 1)
            # id_image = cv2.warpAffine(id_image, M_1, (w, h))
            # id_image = id_image.transpose(2, 0, 1) # HWC => CHW

            with open(label_fname, 'r') as f:
                labels = json.load(f)['labels']

            # load_raw_labels
            labels = dict(labels)
            label = np.array(labels['image'])
            label = label.astype(np.float32)

            return image, label
        except Exception:
            print(f"Errupted data: {image_fname}")
            return None, None, None, None

    @property
    def has_labels(self):
        return True

    @property
    def label_shape(self):
        return [25]

    def __getitem__(self, idx):
        return self._load_all_ffhq(idx)

class FFHQ_GEN_Dataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = 512,  # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path

        self._eg3d_fnames = []
        self._eg3d_path = '/data_14t/data_hzx/zs3nerf/FFHQ_in_the_wlid/'
        if os.path.isdir(self._eg3d_path):
            self._type = 'dir'
            self._eg3d_fnames = glob.glob(os.path.join(self._eg3d_path, r"cropped_image/*.jpg"))
            self._eg3d_fnames.sort()
            self._eg3d_fnames = self._eg3d_fnames[:-8000]
            with open(os.path.join(self._eg3d_path, 'label/labels.json'), 'r') as file:
                self._eg3d_labels = json.load(file)
        else:
            raise IOError('Path must point to a directory')

        self._gen_fnames = []
        if os.path.isdir(self._path):
            self._type = 'dir'
            all_dirs = os.scandir(self._path)
            for i in all_dirs:
                image_path = os.path.join(self._path, i.name, i.name+'_f.jpg')
                if os.path.isfile(image_path):
                    self._gen_fnames.append(image_path)
            self._gen_fnames.sort()
            self._gen_fnames = self._gen_fnames[:6000]
            with open(os.path.join(self._path, 'pose_labels.json'), 'r') as file:
                self._pose_labels = json.load(file)
            self._depth_images = np.load(os.path.join(self._path, 'depth_images.npy'), allow_pickle=True)
        else:
            raise IOError('Path must point to a directory')

        if len(self._gen_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = "eg3d"
        raw_shape = [len(self._eg3d_fnames)] + [3] + [512] * 2
        self._raw_labels_std = torch.zeros((25))
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    @staticmethod
    def aug_image(image, random_s=None, is_loss=False):
        # randomly scale
        image = cv2.resize(np.asarray(image[random_s:-(random_s+1), random_s:-(random_s+1), :]), (512, 512))
        # randomly rotate image
        random_state = np.random.randint(2, size=1)
        if(random_state and not is_loss):
            rotate_angle = np.random.randint(-15, 15)
            M = cv2.getRotationMatrix2D((256,256), rotate_angle, 1)
            image = cv2.warpAffine(image, M, (512,512))
        return image

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    
    def get_label(self, idx):
        image_fname = self._eg3d_fnames[self._raw_idx[idx]]
        label_index = os.path.basename(image_fname).replace('.jpg', '.png')
        label = np.array(self._eg3d_labels[label_index]).astype(np.float32)
        return label

    def _load_gen_all(self, idx):
        output = {}
        image_name = self._gen_fnames[self._raw_idx[idx]]
        condition_image = cv2.imread(image_name.replace('f.jpg', 'f.jpg'))
        condition_image = cv2.cvtColor(condition_image, cv2.COLOR_BGR2RGB)
        condition_image = condition_image.transpose(2, 0, 1) # HWC => CHW
        loss_image = cv2.imread(image_name.replace('f.jpg', 's.jpg'))
        flip_image = cv2.flip(loss_image, 1)
        loss_image = cv2.cvtColor(loss_image, cv2.COLOR_BGR2RGB)
        flip_image = cv2.cvtColor(flip_image, cv2.COLOR_BGR2RGB)
        loss_image = loss_image.transpose(2, 0, 1) # HWC => CHW
        flip_image = flip_image.transpose(2, 0, 1) # HWC => CHW

        # load random image
        random_index = np.random.randint(0, len(self._gen_fnames))
        while random_index == self._raw_idx[idx]:
            random_index = np.random.randint(0, len(self._gen_fnames))
        random_name = self._gen_fnames[random_index]
        random_image = cv2.imread(self._gen_fnames[random_index])
        random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
        random_image = random_image.transpose(2, 0, 1) # HWC => CHW

        output['condition_image'] = condition_image
        output['condition_c'    ] = np.array(self._pose_labels[os.path.basename(image_name).replace('.jpg','.json')]).astype(np.float32)
        output['loss_image'     ] = loss_image
        output['loss_c'         ] = np.array(self._pose_labels[os.path.basename(image_name).replace('.jpg','.json').replace('f', 's')]).astype(np.float32)
        output['random_image'   ] = random_image
        output['random_c'       ] = np.array(self._pose_labels[os.path.basename(random_name).replace('.jpg','.json')]).astype(np.float32)
        output['c_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg',''))
        output['l_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg','').replace('f', 's'))
        output['flip_image'     ] = flip_image
        output['factor'] = 1.0
        
        return output

    def _load_all_eg3d(self, idx):
        output = {}
        image_fname = self._eg3d_fnames[self._raw_idx[idx]]
        label_index = os.path.basename(image_fname).replace('.jpg', '.png')
        image = cv2.imread(image_fname)
        # flip image
        flip_image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        flip_image = cv2.cvtColor(flip_image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1) # HWC => CHW
        flip_image = flip_image.transpose(2, 0, 1) # HWC => CHW
        label = np.array(self._eg3d_labels[label_index]).astype(np.float32)
        
        # load random image
        random_index = np.random.randint(0, len(self._eg3d_fnames))
        while random_index == self._raw_idx[idx]:
            random_index = np.random.randint(0, len(self._eg3d_fnames))
        random_name = self._eg3d_fnames[random_index]
        random_image = cv2.imread(self._eg3d_fnames[random_index])
        random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
        random_image = random_image.transpose(2, 0, 1) # HWC => CHW

        # give a random angle and ignore the image loss
        if np.random.rand() > 0.5:
            output['factor'] = 0.0
            image_fname = self._eg3d_fnames[np.random.randint(0, len(self._gen_fnames))]
            label_index = os.path.basename(image_fname).replace('.jpg', '.png')
            label = np.array(self._eg3d_labels[label_index]).astype(np.float32)
            # pitch_range = 0.3
            # yaw_range = 0.75
            # cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * np.random.random()), 3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * np.random.random()),\
            #                                             torch.tensor([0, 0, 0.2]), radius=2.7)
            # label[:16] = cam2world_pose.numpy().reshape(-1)
        else:
            output['factor'] = 1.0

        output['condition_image'] = image
        output['loss_image'     ] = image
        output['random_image'   ] = random_image
        output['random_c'       ] = np.array(self._eg3d_labels[os.path.basename(random_name).replace('.jpg','.png')]).astype(np.float32)
        output['loss_c'         ] = label
        output['flip_image'         ] = flip_image

        idx = np.random.randint(0, len(self._gen_fnames))
        image_name = self._gen_fnames[idx]
        output['condition_c'    ] = np.array(self._pose_labels[os.path.basename(image_name).replace('.jpg','.json')]).astype(np.float32)
        output['c_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg',''))
        output['l_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg','').replace('f', 's'))

        return output
    
    @property
    def has_labels(self):
        return True

    @property
    def label_shape(self):
        return [25]

    def __getitem__(self, idx):
        if idx < len(self._gen_fnames) - 1:
            if np.random.rand(1) > 0.5:
                return self._load_gen_all(idx)
        return self._load_all_eg3d(idx)
    
class Test_Dataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = 512,  # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path

        self._eg3d_fnames = []
        self._eg3d_path = '/data_14t/data_hzx/zs3nerf/FFHQ_in_the_wlid/'
        if os.path.isdir(self._eg3d_path):
            self._type = 'dir'
            self._eg3d_fnames = glob.glob(os.path.join(self._eg3d_path, r"cropped_image/*.jpg"))
            self._eg3d_fnames.sort()
            self._eg3d_fnames = self._eg3d_fnames[-8000:]
            with open(os.path.join(self._eg3d_path, 'label/labels.json'), 'r') as file:
                self._eg3d_labels = json.load(file)
        else:
            raise IOError('Path must point to a directory')

        name = "eg3d"
        raw_shape = [len(self._eg3d_fnames)] + [3] + [512] * 2
        self._raw_labels_std = torch.zeros((25))
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    
    def get_label(self, idx):
        image_fname = self._eg3d_fnames[self._raw_idx[idx]]
        label_index = os.path.basename(image_fname).replace('.jpg', '.png')
        label = np.array(self._eg3d_labels[label_index]).astype(np.float32)
        return label

    def _load_all_eg3d(self, idx):
        output = {}
        image_fname = self._eg3d_fnames[self._raw_idx[idx]]
        label_index = os.path.basename(image_fname).replace('.jpg', '.png')
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1) # HWC => CHW
        label = np.array(self._eg3d_labels[label_index]).astype(np.float32)

        output['condition_image'] = image
        output['loss_image'     ] = image
        output['condition_c'    ] = label
        output['loss_c'         ] = label

        return output
    
    @property
    def has_labels(self):
        return True

    @property
    def label_shape(self):
        return [25]

    def __getitem__(self, idx):
        return self._load_all_eg3d(idx)
    
class Afhqv2_Dataset(Dataset):
    def __init__(self,
        path            = None,                   # Path to directory or zip.
        resolution      = 512,  # Ensure specific resolution, None = highest available.
        name            = None,
        raw_shape       = None,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        if name is not None:
            super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
            return
        self._path = path

        self._afhqv2_fnames = []
        self._afhqv2_path = '/home/huangzixiong/dataset/classnerf/afhq_v2'
        if os.path.isdir(self._afhqv2_path):
            self._type = 'dir'
            self._afhqv2_fnames = glob.glob(os.path.join(self._afhqv2_path, r"train/cat/*.png"))
            self._afhqv2_fnames.sort()
            self._afhqv2_fnames = self._afhqv2_fnames[:4000]
            with open(os.path.join(self._afhqv2_path, 'train/label/labels.json'), 'r') as file:
                self._afhqv2_labels = json.load(file)
        else:
            raise IOError('Path must point to a directory')

        self._gen_fnames = []
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._gen_fnames = glob.glob(os.path.join(self._path, r"*/*_f.jpg"))
            self._gen_fnames.sort()
            with open(os.path.join(self._path, 'pose_labels.json'), 'r') as file:
                self._pose_labels = json.load(file)
            self._depth_images = np.load(os.path.join(self._path, 'depth_images.npy'), allow_pickle=True)
        else:
            raise IOError('Path must point to a directory')

        if len(self._gen_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = "afhqv2"
        raw_shape = [len(self._afhqv2_fnames)] + [3] + [512] * 2
        self._raw_labels_std = torch.zeros((25))
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    
    def get_label(self, idx):
        image_fname = self._afhqv2_fnames[self._raw_idx[idx]]
        label_index = os.path.basename(image_fname)
        label = np.array(self._afhqv2_labels[label_index]).astype(np.float32)
        return label

    def _load_all_Afhqv2(self, idx):
        output = {}
        image_fname = self._afhqv2_fnames[self._raw_idx[idx]]
        label_index = os.path.basename(image_fname)
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1) # HWC => CHW
        label = np.array(self._afhqv2_labels[label_index]).astype(np.float32)

        # load random image
        random_index = np.random.randint(0, len(self._afhqv2_fnames))
        while random_index == self._raw_idx[idx]:
            random_index = np.random.randint(0, len(self._afhqv2_fnames))
        random_image = cv2.imread(self._afhqv2_fnames[random_index])
        random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
        random_image = random_image.transpose(2, 0, 1) # HWC => CHW

        # give a random angle and ignore the image loss
        if np.random.rand() > 0.5:
            output['factor'] = 0.0
            image_fname = self._gen_fnames[np.random.randint(0, len(self._gen_fnames))]
            # label_index = os.path.basename(image_fname).replace('.jpg', '.png')
            label = np.array(self._pose_labels[os.path.basename(image_fname).replace('.jpg','.json')]).astype(np.float32)
        else:
            output['factor'] = 1.0

        output['condition_image'] = image
        output['loss_image'     ] = image
        output['random_image'   ] = random_image
        output['loss_c'         ] = label

        image_name = self._gen_fnames[self._raw_idx[idx]]
        output['condition_c'    ] = np.array(self._pose_labels[os.path.basename(image_name).replace('.jpg','.json')]).astype(np.float32)
        output['c_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg',''))
        output['l_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg','').replace('f', 's'))

        return output

    def _load_gen_all(self, idx):
        output = {}
        image_name = self._gen_fnames[self._raw_idx[idx]]
        # load_raw_image
        # print(image_name.replace('f.jpg', 'f_mask.jpg'))
        condition_image = cv2.imread(image_name.replace('f.jpg', 'f.jpg'))
        # random_s = np.random.randint(0, 30)
        # condition_image = self.aug_image(condition_image, random_s)
        condition_image = cv2.cvtColor(condition_image, cv2.COLOR_BGR2RGB)
        condition_image = condition_image.transpose(2, 0, 1) # HWC => CHW
        loss_image = cv2.imread(image_name.replace('f.jpg', 's.jpg'))
        # loss_image = self.aug_image(loss_image, random_s, True)
        loss_image = cv2.cvtColor(loss_image, cv2.COLOR_BGR2RGB)
        loss_image = loss_image.transpose(2, 0, 1) # HWC => CHW

        # load random image
        random_index = np.random.randint(0, len(self._gen_fnames))
        while random_index == self._raw_idx[idx]:
            random_index = np.random.randint(0, len(self._gen_fnames))
        random_name = self._gen_fnames[random_index]
        random_image = cv2.imread(random_name)
        random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
        random_image = random_image.transpose(2, 0, 1) # HWC => CHW

        output['condition_image'] = condition_image
        output['condition_c'    ] = np.array(self._pose_labels[os.path.basename(image_name).replace('.jpg','.json')]).astype(np.float32)
        output['loss_image'     ] = loss_image
        output['loss_c'         ] = np.array(self._pose_labels[os.path.basename(image_name).replace('.jpg','.json').replace('f', 's')]).astype(np.float32)
        output['random_image'   ] = random_image
        output['c_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg',''))
        output['l_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg','').replace('f', 's'))

        output['factor'] = 1.0
        
        return output
    
    @property
    def has_labels(self):
        return True

    @property
    def label_shape(self):
        return [25]

    def __getitem__(self, idx):
        if idx < len(self._afhqv2_fnames) - 1:
            if np.random.rand(1) > 0.5:
                return self._load_all_Afhqv2(idx)
        return self._load_gen_all(idx)
    
class Afhqv2_Test_Dataset(Afhqv2_Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = 512,  # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path

        self._afhqv2_fnames = []
        self._afhqv2_path = '/home/huangzixiong/dataset/classnerf/afhq_v2'
        if os.path.isdir(self._afhqv2_path):
            self._type = 'dir'
            self._afhqv2_fnames = glob.glob(os.path.join(self._afhqv2_path, r"train/cat/*.png"))
            self._afhqv2_fnames.sort()
            self._afhqv2_fnames = self._afhqv2_fnames[4000:]
            with open(os.path.join(self._afhqv2_path, 'train/label/labels.json'), 'r') as file:
                self._afhqv2_labels = json.load(file)
        else:
            raise IOError('Path must point to a directory')

        name = "afhqv2"
        raw_shape = [len(self._afhqv2_fnames)] + [3] + [512] * 2
        self._raw_labels_std = torch.zeros((25))
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    
    def get_label(self, idx):
        image_fname = self._afhqv2_fnames[self._raw_idx[idx]]
        label_index = os.path.basename(image_fname).replace('.jpg', '.png')
        label = np.array(self._afhqv2_fnames[label_index]).astype(np.float32)
        return label

    def _load_all_eg3d(self, idx):
        output = {}
        image_fname = self._afhqv2_fnames[self._raw_idx[idx]]
        label_index = os.path.basename(image_fname).replace('.jpg', '.png')
        image = cv2.imread(image_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1) # HWC => CHW
        label = np.array(self._afhqv2_labels[label_index]).astype(np.float32)

        output['condition_image'] = image
        output['loss_image'     ] = image
        output['condition_c'    ] = label
        output['loss_c'         ] = label

        return output
    
    @property
    def has_labels(self):
        return True

    @property
    def label_shape(self):
        return [25]

    def __getitem__(self, idx):
        return self._load_all_eg3d(idx)

class ShapeNet_Dataset(Dataset):
    def __init__(self,
        path            = None,                   # Path to directory or zip.
        resolution      = 128,  # Ensure specific resolution, None = highest available.
        name            = None,
        raw_shape       = None,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        if name is not None:
            super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
            return
        self._path = path

        self._shapenet_fnames = []
        self._shapenet_path = '/home/huangzixiong/dataset/classnerf/srn_chairs/chairs_train'
        if os.path.isdir(self._shapenet_path):
            self._type = 'dir'
            with open(os.path.join(self._shapenet_path, 'train_up_sphere.txt'), 'r') as file:
                self._shapenet_fnames = file.readlines()
                self._shapenet_fnames = [i.replace('\n', '') for i in self._shapenet_fnames]
            self._shapenet_fnames.sort()
            with open(os.path.join(self._shapenet_path, 'label/labels.json'), 'r') as file:
                self._shapenet_labels = json.load(file)
        else:
            raise IOError('Path must point to a directory')

        self._gen_fnames = []
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._gen_fnames = glob.glob(os.path.join(self._path, r"*/*_f.jpg"))
            self._gen_fnames.sort()
            with open(os.path.join(self._path, 'pose_labels.json'), 'r') as file:
                self._pose_labels = json.load(file)
            self._depth_images = np.load(os.path.join(self._path, 'depth_images.npy'), allow_pickle=True)
        else:
            raise IOError('Path must point to a directory')

        if len(self._gen_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = "shapenet"
        raw_shape = [len(self._shapenet_fnames)] + [3] + [resolution] * 2
        self._raw_labels_std = torch.zeros((25))
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    
    def get_label(self, idx):
        image_fname = self._shapenet_fnames[self._raw_idx[idx]]
        label_index = image_fname
        label = np.array(self._shapenet_labels[label_index]).astype(np.float32)
        return label

    def _load_all_ShapeNet(self, idx):
        output = {}
        image_fname = self._shapenet_fnames[self._raw_idx[idx]]
        label_index = image_fname
        image = cv2.imread(os.path.join(self._shapenet_path, image_fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1) # HWC => CHW
        label = np.array(self._shapenet_labels[label_index]).astype(np.float32)

        # load random image
        random_index = np.random.randint(0, len(self._shapenet_fnames))
        while random_index == self._raw_idx[idx]:
            random_index = np.random.randint(0, len(self._shapenet_fnames))
        random_image = cv2.imread(os.path.join(self._shapenet_path, self._shapenet_fnames[random_index]))
        random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
        random_image = random_image.transpose(2, 0, 1) # HWC => CHW

        # give a random angle and ignore the image loss
        if np.random.rand() > 0.5:
            output['factor'] = 0.0
            image_fname = self._gen_fnames[np.random.randint(0, len(self._gen_fnames))]
            # label_index = os.path.basename(image_fname).replace('.jpg', '.png')
            label = np.array(self._pose_labels[os.path.basename(image_fname).replace('.jpg','.json')]).astype(np.float32)
        else:
            output['factor'] = 1.0

        output['condition_image'] = image
        output['loss_image'     ] = image
        output['random_image'   ] = random_image
        output['loss_c'         ] = label

        image_name = self._gen_fnames[np.random.randint(0, len(self._gen_fnames))]
        # image_name = self._gen_fnames[self._raw_idx[idx]]
        output['condition_c'    ] = np.array(self._pose_labels[os.path.basename(image_name).replace('.jpg','.json')]).astype(np.float32)
        output['c_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg',''))
        output['l_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg','').replace('f', 's'))

        return output

    def _load_gen_all(self, idx):
        output = {}
        image_name = self._gen_fnames[idx]
        # load_raw_image
        condition_image = cv2.imread(image_name.replace('f.jpg', 'f.jpg'))
        condition_image = cv2.cvtColor(condition_image, cv2.COLOR_BGR2RGB)
        condition_image = condition_image.transpose(2, 0, 1) # HWC => CHW
        loss_image = cv2.imread(image_name.replace('f.jpg', 's.jpg'))
        loss_image = cv2.cvtColor(loss_image, cv2.COLOR_BGR2RGB)
        loss_image = loss_image.transpose(2, 0, 1) # HWC => CHW

        # load random image
        random_index = np.random.randint(0, len(self._gen_fnames))
        while random_index == self._raw_idx[idx]:
            random_index = np.random.randint(0, len(self._gen_fnames))
        random_name = self._gen_fnames[random_index]
        random_image = cv2.imread(random_name)
        random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)
        random_image = random_image.transpose(2, 0, 1) # HWC => CHW

        output['condition_image'] = condition_image
        output['condition_c'    ] = np.array(self._pose_labels[os.path.basename(image_name).replace('.jpg','.json')]).astype(np.float32)
        output['loss_image'     ] = loss_image
        output['loss_c'         ] = np.array(self._pose_labels[os.path.basename(image_name).replace('.jpg','.json').replace('f', 's')]).astype(np.float32)
        output['random_image'   ] = random_image
        output['c_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg',''))
        output['l_depth_image'  ] = self._depth_images.item().get(os.path.basename(image_name).replace('.jpg','').replace('f', 's'))

        output['factor'] = 1.0
        
        return output
    
    @property
    def has_labels(self):
        return True

    @property
    def label_shape(self):
        return [25]

    def __getitem__(self, idx):
        if idx < len(self._gen_fnames) - 1:
            if np.random.rand(1) > 0.5:
                return self._load_gen_all(idx)
        return self._load_all_ShapeNet(idx)
    
class ShapeNet_Test_Dataset(Afhqv2_Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = 128,  # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path

        self._shapenet_fnames = []
        self._shapenet_path = '/home/huangzixiong/dataset/classnerf/srn_chairs/chairs_test'
        if os.path.isdir(self._shapenet_path):
            self._type = 'dir'
            with open(os.path.join(self._shapenet_path, 'train.txt'), 'r') as file:
                self._shapenet_fnames = file.readlines()
                self._shapenet_fnames = [i.replace('\n', '') for i in self._shapenet_fnames]
                self._shapenet_fnames.sort()
            with open(os.path.join(self._shapenet_path, 'label/labels.json'), 'r') as file:
                self._shapenet_labels = json.load(file)
        else:
            raise IOError('Path must point to a directory')

        name = "ShapeNet"
        raw_shape = [len(self._shapenet_fnames)] + [3] + [resolution] * 2
        self._raw_labels_std = torch.zeros((25))
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    
    def get_label(self, idx):
        image_fname = self._shapenet_fnames[self._raw_idx[idx]]
        label_index = image_fname
        label = np.array(self._shapenet_fnames[label_index]).astype(np.float32)
        return label

    def _load_all_eg3d(self, idx):
        output = {}
        image_fname = self._shapenet_fnames[self._raw_idx[idx]]
        label_index = image_fname
        image = cv2.imread(os.path.join(self._shapenet_path, image_fname))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1) # HWC => CHW
        label = np.array(self._shapenet_labels[label_index]).astype(np.float32)

        output['condition_image'] = image
        output['loss_image'     ] = image
        output['condition_c'    ] = label
        output['loss_c'         ] = label

        return output
    
    @property
    def has_labels(self):
        return True

    @property
    def label_shape(self):
        return [25]

    def __getitem__(self, idx):
        return self._load_all_eg3d(idx)

if __name__ == '__main__':
    dataloader = ShapeNet_Dataset("/home/huangzixiong/dataset/classnerf/ShapeNet_GEN_DATASET/ShapeNet_GEN_W_0.5/", 128)
    training_set_iterator = torch.utils.data.DataLoader(dataset=dataloader, batch_size=1, num_workers=0)
    import tqdm
    for i in tqdm.tqdm(training_set_iterator):
        print(i['condition_image'].shape)
        print(i['loss_image'].shape)
        print(i['loss_c'].shape)
        print("------------------------------------------------")
        continue
    # dataloader = FFHQ_GEN_Dataset("/mnt/cephfs/dataset/Face/EG3D_GEN", 512)
    # training_set_iterator = torch.utils.data.DataLoader(dataset=dataloader, batch_size=10, num_workers=2)
    # import tqdm
    # for i, l in tqdm.tqdm(training_set_iterator):
    #     print(i.shape)