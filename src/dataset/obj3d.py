from torch.utils.data import Dataset
import os.path as osp
from torchvision import transforms
from skimage import io
import os
import numpy as np
import torch
from PIL import Image, ImageFile
import lmdb
import pickle
import logging

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Obj3D(Dataset):
    def __init__(self, cfg, type, mode):
        self.type = type
        self.img_h = 128
        self.img_w = 128
        self.state = mode
        if type == "small":
            self.obj3d_space_dir = cfg.dataset_roots.OBJ3D_SMALL
        elif type == "large":
            self.obj3d_space_dir = cfg.dataset_roots.OBJ3D_LARGE
        else:
            raise ValueError("type argument can only be small and large")

        if self.state != "train" and self.state != "test" and self.state != "val":
            raise ValueError("state argument can only be train, test and val")

        self.img_transform = transforms.Compose(
        [
            transforms.ToTensor()
        ])

        self.env = lmdb.open(os.path.join(self.obj3d_space_dir, f"obj3d_space_{type}_lmdb"), max_readers=16,
                             readonly=True,
                             lock=False)
        with open(os.path.join(self.obj3d_space_dir, f"obj3d_space_{type}_scene_props_{self.state}.p"), "rb") as f:
            self.scene_props = pickle.load(f)

        if not self.env:
            logging.error("Cannot open lmdb on : " + os.path.join(self.clevr_dir, "clevr_deepmind_lmdb"))
            raise FileNotFoundError()
        
    @property
    def bb_path(self):
        # path = osp.join(self.root, self.mode, 'bb')
        # assert osp.exists(path), f'Bounding box path {path} does not exist.'
        return ""

    def __del__(self):
        try:
            self.env.close()
        except:
            logging.warning("lmdb can not be closed")
    
    def __getitem__(self, idx):
        bboxs = self.scene_props[self.state + "_" + str(idx)]
        with self.env.begin(write=False) as txn:
            img = Image.frombytes("RGB", (self.img_w, self.img_h), txn.get((self.state + "_" + str(idx)).encode()))
            if self.img_transform is not None:
                img = self.img_transform(img)

        if self.state != "train":
            return img, bboxs
        else:
            return img

    def collate_fn(self, batch):
        img_tensor_list = []
        if self.state != "train":
            bboxs_list = []
            for img_tensor, bboxs in batch:
                img_tensor_list.append(img_tensor)
                bboxs_list.append(bboxs)
            return torch.stack(img_tensor_list), [], bboxs_list

        for img_tensor in batch:
            img_tensor_list.append(img_tensor)
        return [torch.stack(img_tensor_list)]
    
    def __len__(self):
        return len(self.scene_props)
    

