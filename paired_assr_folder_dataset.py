# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
from mmedit.datasets.base_sr_dataset import BaseSRDataset
from mmedit.datasets.registry import DATASETS

@DATASETS.register_module()
class PairedASSRFolderDataset(BaseSRDataset):
    """General paired image folder dataset for image restoration.

    The dataset loads lq (Low Quality) and gt (Ground-Truth) image pairs,
    applies specified transforms and finally returns a dict containing paired
    data and other information.

    This is the "folder mode", which needs to specify the lq folder path and gt
    folder path, each folder containing the corresponding images.
    Image lists will be generated automatically. You can also specify the
    filename template to match the lq and gt pairs.

    For example, we have two folders with the following structures:

    ::
        data_root
        ├── lq
        │   ├── 0001
        |   │   ├── 4.png
        |   │   ├── 3.png
        |   │   ├── 2.png
        │   ├── 0002
        |   │   ├── 4.5.png
        |   │   ├── 3.5.png
        |   │   ├── 2.png
        ├── gt
        │   ├── 0001.png
        │   ├── 0002.png

    then, you need to set:
    .. code-block:: python
        lq_folder = data_root/lq
        gt_folder = data_root/gt

    Args:
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """
    def __init__(self,
                 lq_folder,
                 gt_folder,
                 pipeline,
                 scale = None, 
                 roundnum = 0,
                 test_mode=False):
        super().__init__(pipeline, scale, test_mode)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.roundnum = roundnum
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        """Load annoations for SR dataset.
        It loads the LQ and GT image path from folders.
        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        data_infos = []
        lq_paths = os.listdir(self.lq_folder)
        gt_paths = os.listdir(self.gt_folder)
        assert len(lq_paths) == len(gt_paths), (
            f'gt and lq datasets have different number of images: '
            f'{len(lq_paths)}, {len(gt_paths)}.')
        for gt_path in gt_paths:
            hr_img = osp.join(self.gt_folder, gt_path)
            
            basename, ext = osp.splitext(gt_path)
            lr_fold = osp.join(self.lq_folder, basename)
            assert basename in lq_paths, f'{basename} is not in lq_paths.'
            
            for lq_path in os.listdir(lr_fold):
                if self.scale is not None:
                    basename, ext = osp.splitext(lq_path)
                    scale = round(float(basename), self.roundnum)
                    if scale == self.scale:
                        data_infos.append(dict(lq_path=osp.join(lr_fold, lq_path), gt_path=hr_img))
                else:
                    data_infos.append(dict(lq_path=osp.join(lr_fold, lq_path), gt_path=hr_img))
                
        return data_infos

