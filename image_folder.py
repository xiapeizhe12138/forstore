import os
from PIL import Image
from io import BytesIO
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from loadutils import FileClient
# from datasets import register
import pillow_avif
import lmdb
import math

def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.
    """

    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def aligncrop(hr, lr, patch):
    _, H, W = hr.shape
    _, h, w = lr.shape
    rh = H/h
    rw = W/w
    h_lr_start = random.randint(0, h-patch)#0#
    w_lr_start = random.randint(0, w-patch)#0#
    lrcrop = lr[:, h_lr_start:h_lr_start+patch, w_lr_start:w_lr_start+patch]
    
    h_lr_start = h_lr_start*rh
    h_hr_start = math.ceil(h_lr_start)
    w_lr_start = w_lr_start*rw
    w_hr_start = math.ceil(w_lr_start)
    h_lr_end = (h_lr_start+patch)*rh
    h_hr_end = math.floor(h_lr_end)
    w_lr_end = (w_lr_start+patch)*rw
    w_hr_end = math.floor(w_lr_end)
    lrcrophr = hr[:, h_hr_start:h_hr_end, w_hr_start:w_hr_end]
    
    cropside = ((h_hr_start-h_lr_start,
                 h_lr_end - h_hr_end),
                 (w_hr_start-w_lr_start,
                 w_lr_end - w_hr_end))

    hr_coord_range = ((cropside[0][0]*2/patch/rh-1, 
                       1-cropside[0][1]*2/patch/rh), 
                      (cropside[1][0]*2/patch/rw-1, 
                       1-cropside[1][1]*2/patch/rw))
    hr_coord = make_coord(lrcrophr.shape[-2:], ranges=hr_coord_range, flatten=False)#True)
    
    return lrcrop, lrcrophr, hr_coord, rh, rw

def list_all_keys(lmdb_path):
    """
    列出LMDB数据库中的所有key。
    
    :param lmdb_path: LMDB数据库的文件路径。
    :return: key列表
    """
    # 打开LMDB环境
    env = lmdb.open(lmdb_path)
    keys = []

    # 遍历LMDB中的键值对，提取键
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            keys.append(key.decode('utf-8'))  # 解码key并添加到列表中

    env.close()
    return keys


# @register('image-folder')
class ImageFolderlmdb(Dataset):

    def __init__(self, lr_path, gt_path, repeat=1, scanfold=True):
        self.repeat = repeat
        self.lr_lmdbpath = []
        self.lr_imgpath = {}
        self.lr_img = []
        self.file_client = None
        self.io_backend_opt = {}
        
        if scanfold:
            for dir in os.listdir(lr_path):
                lmdbpath = os.path.join(os.path.join(lr_path, dir))
                self.lr_lmdbpath.append(lmdbpath)
        else:
            self.lr_lmdbpath.append(lmdbpath)
        
        for lmdb_path in self.lr_lmdbpath:
            keys = list_all_keys(lmdb_path)
            for key in keys:
                self.lr_imgpath[key] = lmdb_path
                self.lr_img.append(key)

        self.io_backend_opt['type'] = 'lmdb'
        self.io_backend_opt['db_paths'] = self.lr_lmdbpath.copy()
        self.io_backend_opt['client_keys'] = self.lr_lmdbpath.copy()
        
        self.io_backend_opt['db_paths'].append(gt_path)
        self.io_backend_opt['client_keys'].append('gt')

    def __len__(self):
        return len(self.lr_imgpath) * self.repeat

    def __getitem__(self, idx):
        lr_key = self.lr_img[idx % len(self.lr_imgpath)]
        lr_lmdb = self.lr_imgpath[lr_key]
        hr_key = lr_key[:7]+'.png'
        
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
            
        img_bytes = self.file_client.get(lr_key, lr_lmdb)
        lr_image = Image.open(BytesIO(img_bytes))
        lr_image = lr_image.convert('RGB')
        
        img_bytes = self.file_client.get(hr_key, 'gt')
        hr_image = Image.open(BytesIO(img_bytes))
        hr_image = hr_image.convert('RGB')
        
        to_tensor_transform = transforms.ToTensor()
        lr_image = to_tensor_transform(lr_image)
        hr_image = to_tensor_transform(hr_image)
        
        return lr_image, hr_image

# @register('sr-implicit-paired')
class SRcropPaired(Dataset):

    def __init__(self, dataset, patch=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.patch = patch
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        
        if self.patch==None:
            crop_lr, crop_hr = img_lr, img_hr
            rh = crop_hr.shape[-2]/crop_lr.shape[-2]
            rw = crop_hr.shape[-1]/crop_lr.shape[-1]
            hr_coord = make_coord(crop_hr.shape[-2:], flatten=False)
        else:
            crop_lr, crop_hr, hr_coord, rh, rw = aligncrop(img_hr, img_lr, self.patch)
        # assert rh == rw, 'no shape change please'
        assert hr_coord.dim()==3, 'hr_coord.dim()!=3'
        if self.augment:
            if random.random() < 0.5:
                crop_lr = crop_lr.flip(-2)
                crop_hr = crop_hr.flip(-2)
                hr_coord = hr_coord.flip(0)#coord的每一个信息点要与hr相对应
                hr_coord[:, :, 0] = hr_coord[:, :, 0]*(-1)#维度反转对coord对应维度影响
            if random.random() < 0.5:
                crop_lr = crop_lr.flip(-1)
                crop_hr = crop_hr.flip(-1)
                hr_coord = hr_coord.flip(1)
                hr_coord[:, :, 1] = hr_coord[:, :, 1]*(-1)
            if random.random() < 0.5:
                crop_lr = crop_lr.transpose(-2, -1)
                crop_hr = crop_hr.transpose(-2, -1)
                hr_coord = hr_coord.transpose(0, 1)
                hr_coord = hr_coord.flip(-1)

        crop_hr = crop_hr.view(3,-1).permute(1,0).contiguous()
        hr_coord = hr_coord.view(-1,2)
        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / (rh * crop_lr.shape[-2])
        cell[:, 1] *= 2 / (rw * crop_lr.shape[-1])

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


def save_tensor_as_image(tensor, save_path):
    print(tensor.shape)
    if tensor.ndimension() != 3 or tensor.size(0) != 3:
        raise ValueError("Expected a [C, H, W] tensor with 3 channels (RGB).")

    transform = transforms.ToPILImage()
    img = transform(tensor)

    img.save(save_path, format='PNG')
    print(f"Image saved to {save_path}")
lr_path = r'lr'
lenth = 0
for dir in os.listdir(lr_path):
    lmdbpath = os.path.join(os.path.join(lr_path, dir))
    keys = list_all_keys(lmdbpath)
    print(dir)
    print(len(keys))
    print()
    
    lenth += len(keys)
print(lenth)