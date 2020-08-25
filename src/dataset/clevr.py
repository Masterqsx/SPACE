from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import os
import numpy as np
import torch
from PIL import Image, ImageFile
import logging
import lmdb
import pickle
from torch.nn import functional as F
import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CLEVR_DEEPMIND_Crop:
    def __call__(self, sample):
        # Assuming the sample single image is 3D tensor, even for greyscale image
        crop_img = sample[:, 29:221, 64:256].unsqueeze(dim=0)
        crop_img = F.interpolate(crop_img, size=(128, 128), mode='bilinear', align_corners=True)[0]
        if crop_img.shape[0] == 1:
            crop_img[crop_img > 0.] = 1.
        return crop_img

class CLEVR(Dataset):
    def __init__(self, cfg, mode):
        # path = os.path.join(root, mode)
        self.clevr_dir = cfg.dataset_roots.CLEVR
        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor()
                , CLEVR_DEEPMIND_Crop()
            ])
        self.mask_transform = transforms.Compose(
            [
                transforms.ToTensor()
                , CLEVR_DEEPMIND_Crop()
            ])
        self.fixed_mask_len = 7
        self.env = lmdb.open(os.path.join(self.clevr_dir, "clevr_deepmind_lmdb"), max_readers=16, readonly=True,
                             lock=False)

        if mode == 'train':
            type = "clevr6_train0_bbox"
        else:
            type = "clevr6_test0_bbox"

        if type == "clevr":
            with open(os.path.join(self.clevr_dir, "clevr_deepmind_scene_props.p"), "rb") as f:
                self.scene_props = pickle.load(f)
        elif type == "clevr6":
            with open(os.path.join(self.clevr_dir, "clevr6_deepmind_scene_props.p"), "rb") as f:
                self.scene_props = pickle.load(f)
        elif type == "clevr6_train0":
            with open(os.path.join(self.clevr_dir, "clevr6_deepmind_trainset_scene_props_0.p"), "rb") as f:
                self.scene_props = pickle.load(f)
        elif type == "clevr6_test0":
            with open(os.path.join(self.clevr_dir, "clevr6_deepmind_testset_scene_props_0.p"), "rb") as f:
                self.scene_props = pickle.load(f)
        elif type == "clevr6_train0_bbox":
            with open(os.path.join(self.clevr_dir, "clevr6_deepmind_trainset_bbox_scene_props_0.p"), "rb") as f:
                self.scene_props = pickle.load(f)
        elif type == "clevr6_test0_bbox":
            with open(os.path.join(self.clevr_dir, "clevr6_deepmind_testset_bbox_scene_props_0.p"), "rb") as f:
                self.scene_props = pickle.load(f)

        if not self.env:
            logging.error("Cannot open lmdb on : " + os.path.join(self.clevr_dir, "clevr_deepmind_lmdb"))
            raise FileNotFoundError()
        self.has_bbox = True

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
        object_props = self.scene_props[idx]
        image_id = object_props[0]["image_id"]
        with self.env.begin(write=False) as txn:
            img = Image.frombytes('RGB', (320, 240), txn.get(("image" + str(image_id)).encode()))
            if self.img_transform is not None:
                img = self.img_transform(img)
            masks = []
            if self.has_bbox:
                bboxs = []
            for object_idx in range(len(object_props)):
                if self.has_bbox:
                    bboxs.append(object_props[object_idx]["bounding_box"])
                obj_id = object_props[object_idx]["object_id"]
                mask = Image.frombytes('L', (320, 240), txn.get(("mask" + str(image_id) + "_" + str(obj_id)).encode()))
                if self.mask_transform is not None:
                    mask = self.mask_transform(mask)
                masks.append(mask[0])

            # Add dummy mask for ARI metrics as pytorch tensor
            # Must be used with ToTensor in pytorch transform
            if self.fixed_mask_len is not None and self.fixed_mask_len > len(object_props):
                for dummy_mask_idx in range(len(object_props), self.fixed_mask_len):
                    masks.append(torch.zeros(128, 128))
                    if self.has_bbox:
                        bboxs.append((-1, -1, -1, -1))
        if self.has_bbox:
            return img, masks, bboxs
        return img, masks
        
    def __len__(self):
        return len(self.scene_props)

    def collate_fn(self, batch):
        img_tensor_list = []
        masks_list = []
        if self.has_bbox:
            bboxs_list = []
            for img_tensor, masks, bboxs in batch:
                img_tensor_list.append(img_tensor)
                masks_list.append(torch.stack(masks))
                bboxs_list.append(bboxs)
            return torch.stack(img_tensor_list), masks_list, bboxs_list

        for img_tensor, masks in batch:
            img_tensor_list.append(img_tensor)
            masks_list.append(torch.stack(masks))
        return torch.stack(img_tensor_list), masks_list

    def sep(self, img):
        """
        Seperate a color image into masks
        :img: (H, W, 3)
        :return: (K, H, W), bool array
        """
        img = img[:, :, :3]
        # a = set()
        # for i in range(img.shape[0]):
        #     for j in range(img.shape[1]):
        #         pixel = tuple(img[i, j][:3])
        #         if pixel not in a and pixel != (64, 64, 64):
        #             a.add(pixel)
        H, W, _ = img.shape
        pixels = list(tuple(pix) for pix in img.reshape(H * W, 3))
        a = set(pixels)
        # background
        a.remove((64, 64, 64))
        masks = []
        for pixel in a:
            pixel = np.array(pixel)
            # (h, w, 3)
            mask = img == pixel
            mask = np.all(mask, 2)
            masks.append(mask)
    
        return masks
    
