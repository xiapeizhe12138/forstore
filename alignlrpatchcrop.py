import math
import random
import numpy as np
import torch

from mmedit.datasets.registry import PIPELINES
from mmedit.datasets.pipelines.utils import make_coord

@PIPELINES.register_module()
class AlignLRPatchCrop:
    """Generate croped LQ and GT image with aligned coord.

    Args:
        patch_size (int): The cropped lr patch size.
            Default: None, means no crop.
    """
    def __init__(self, patch_size):
        self.patch_size = patch_size
        
    def __call__(self, results):
        """Call function.
        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation. 'gt' is required.
        Returns:
            dict: A dict containing the processed data and information.
                modified 'gt', supplement 'lq' and 'scale' to keys.
        """

        gt = results['gt']
        lr = results['lq']
        H, W = gt.shape[-3], gt.shape[-2]
        h, w = lr.shape[-3], lr.shape[-2]
        rh = H/h
        rw = W/w

        #full lr, lr grid, lenth(h,w)=lr(h,w), cell: 1, 1
        h_lr_start = random.randint(0, h-self.patch_size)#0#
        w_lr_start = random.randint(0, w-self.patch_size)#0#
        lrcrop = lr[h_lr_start:h_lr_start+self.patch_size, w_lr_start:w_lr_start+self.patch_size, :]

        #full gt, hr grid, lenth(h,w)=gt(h,w), cell: 1, 1
        h_lr_start = h_lr_start*rh
        h_gt_start = math.ceil(h_lr_start)
        w_lr_start = w_lr_start*rw
        w_gt_start = math.ceil(w_lr_start)
        h_lr_end = (h_lr_start+self.patch_size)*rh
        h_gt_end = math.floor(h_lr_end)
        w_lr_end = (w_lr_start+self.patch_size)*rw
        w_gt_end = math.floor(w_lr_end)
        lrcropgt = gt[h_gt_start:h_gt_end, w_gt_start:w_gt_end, :]

        #full gt, patch in hr grid, lenth(h,w)=gt(h,w), cell: 1, 1
        cropside = ((h_gt_start-h_lr_start,
                    h_lr_end - h_gt_end),
                    (w_gt_start-w_lr_start,
                    w_lr_end - w_gt_end))
        
        #lrpatch, patch in hr grid, lenth(h,w)=(1-(-1), 1-(-1)) 
        gt_coord_range = ((cropside[0][0]*2/self.patch_size/rh-1, 
                        1-cropside[0][1]*2/self.patch_size/rh), 
                        (cropside[1][0]*2/self.patch_size/rw-1, 
                        1-cropside[1][1]*2/self.patch_size/rw))
        gt_coord = make_coord(lrcropgt.shape[:2], ranges=gt_coord_range, flatten=False)

        results['lq'] = lrcrop
        results['gt'] = lrcropgt
        results['coord'] = gt_coord
        results['rh'] = rh
        results['rw'] = rw

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'patch_size={self.patch_size}, ')

        return repr_str