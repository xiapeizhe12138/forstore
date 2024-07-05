exp_name = '001_ciaosr_edsr_div2k'
scale_min, scale_max = 1, 4
val_scale = 4
data_type = 'Urban100'  #TODO {Set5, Set14, BSDS100, Urban100, Manga109}

from mmedited.models.restorers.ciaosr import CiaoSR
from mmedited.models.backbones.sr_backbones.ciaosr_net import LocalImplicitSREDSR


# model settings
model = dict(
    type=CiaoSR,
    generator=dict(
        type=LocalImplicitSREDSR,
        encoder=dict(
            type='EDSR',
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=16),
        imnet_q=dict(
            type='MLPRefiner',
            in_dim=4,
            out_dim=3,
            hidden_list=[256, 256, 256, 256]),
        imnet_k=dict(
            type='MLPRefiner',
            in_dim=64,
            out_dim=64,
            hidden_list=[256, 256, 256, 256]),
        imnet_v=dict(
            type='MLPRefiner',
            in_dim=64,
            out_dim=64,
            hidden_list=[256, 256, 256, 256]),
        feat_unfold=True,
        eval_bsize=30000,
        ),
    rgb_mean=(0.4488, 0.4371, 0.4040),
    rgb_std=(1., 1., 1.),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
if val_scale <= 4:
    test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=val_scale, scale=val_scale, tile=192, tile_overlap=32, convert_to='y') # larger tile is better
else:
    test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=val_scale, scale=val_scale, convert_to='y') # x6, x8, x12 


# dataset settings
train_dataset_type = 'PairedASSRFolderDataset'
val_dataset_type = 'PairedASSRFolderDataset'
test_dataset_type = 'PairedASSRFolderDataset'

train_pipeline = [
    dict(type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb'),
    dict(type='AlignLRPatchCrop',
        patch_size=48),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='GTPatchAndCell', patch_size=48, dim2=True, augment=True),
    dict(type='Collect',
        keys=['lq', 'gt', 'coord', 'cell'],
        meta_keys=['gt_path'])
]
valid_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='GenerateCoordinateAndCell', scale=val_scale),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'coord', 'cell'],
        meta_keys=['gt_path'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='GenerateCoordinateAndCell', scale=val_scale),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'coord', 'cell'],
        meta_keys=['gt_path'])
]

data_dir = "data"
lq_path = f'{data_dir}/Classical/' + data_type + '/LRbicx'+str(val_scale)
gt_path = f'{data_dir}/Classical/' + data_type + '/GTmod12'

data = dict(
    workers_per_gpu=8,
    train_dataloader=dict(samples_per_gpu=16, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=20,
        dataset=dict(type=train_dataset_type,
            lq_folder=train_lq_path,
            gt_folder=train_gt_path, 
            pipeline=train_pipeline,  
            scale=None,
            roundnum=0,)),
    val=dict(type=val_dataset_type,
            lq_folder=val_lq_path,
            gt_folder=val_gt_path, 
            pipeline=valid_pipeline,  
            scale=val_scale,
            roundnum=0,),
    test=dict(type=test_dataset_type,
            lq_folder=test_lq_path,
            gt_folder=test_gt_path, 
            pipeline=test_pipeline,  
            scale=val_scale,
            roundnum=0,))






# optimizer
optimizers = dict(type='Adam', lr=1.e-4)

# learning policy
iter_per_epoch = 1000
total_iters = 1000 * iter_per_epoch
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[200000, 400000, 600000, 800000],
    gamma=0.5)

checkpoint_config = dict(interval=3000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=3000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
visual_config = None

run_dir = './work_dirs'
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'{run_dir}/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
test_checkpoint_path = f'{run_dir}/{exp_name}/latest.pth' # use --checkpoint None to enable this path in testing
