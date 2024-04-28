exp_name = 'm_g3_mlp_multi_100wut_lr1e-4'
scale_min, scale_max = 1, 4
val_scale = 4   #TODO
data_type = 'DIV2K'  #TODO {Set5, Set14, BSDS100, Urban100, Manga109}
btsize_per_gpu = 2
workers_per_gpu = 8
patch_size=48
lr = 1.e-4
val_fre=1
save_fre = val_fre
run_dir = r'/data/xpz120/Ciaosrdata/output/ssm'#r'/output'#

from mmedited.models.restorers.mambaciaosr import MambaCiaoSR
from mmedited.models.backbones.change_ssm.m_g3_mlp_multi import MgobalImplicitSRRDN


# model settings
model = dict(
    type=MambaCiaoSR,
    generator=dict(
        type=MgobalImplicitSRRDN,
        encoder=dict(
            type='RDN',
            in_channels=3,
            out_channels=3,
            mid_channels=64,
            num_blocks=16,
            upscale_factor=4,
            num_layers=8,
            channel_growth=64),
        SSM=dict(
            in_dim=4,#no
            out_channel=3,#no
            d_state=16,
            d_conv=3,
            expand=1.),
        LSSM=dict(
            d_model=64,#no
            out_channel_multi=4,
            sumscan=True,
            windowsize=[3, 5],# 7],
            channelweight=[1.5, 1.5],# 1.5],
            expand=2.),
        imnet=dict(
            type='MLPRefiner',
            in_dim=4,#no
            out_dim=3,
            hidden_list=[256, 256, 256, 256]),
        cropstep=200,
        local_size=2,
        mincrop=patch_size,
        feat_unfold=True,
        eval_bsize=True
        ),
    rgb_mean=(0.4488, 0.4371, 0.4040),
    rgb_std=(1., 1., 1.),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
if val_scale <= 4:
    test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=val_scale, scale=val_scale, tile=192, tile_overlap=32, convert_to='rgb') # larger tile is better
else:
    test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=val_scale, scale=val_scale, convert_to='rgb') # x6, x8, x12 

# dataset settings
train_dataset_type = 'SRFolderGTDataset'
val_dataset_type = 'SRFolderGTDataset'
test_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(
        type='RandomDownSampling',
        scale_min=scale_min,
        scale_max=scale_max,
        patch_size=patch_size),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='GenerateCoordinateAndCell', sample_quantity=patch_size, dim2=True),
    dict(
        type='Collect',
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
    dict(type='RandomDownSampling', scale_min=val_scale, scale_max=val_scale),  
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='GenerateCoordinateAndCell', scale=val_scale, dim2=True),  
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
    dict(type='GenerateCoordinateAndCell', scale=val_scale, dim2=True),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'coord', 'cell'],
        meta_keys=['gt_path'])
]

data_dir = r"/data/xpz120/SRbenchmark"

lq_path = r"/data/xpz120/SRbenchmark/DIV2K/DIV2K_valid_LR_bicubic_X4"
#f'{data_dir}/' + data_type + r'/LR_bicubic/X' + str(val_scale)
gt_path = r"/data/xpz120/SRbenchmark/DIV2K/DIV2K_valid_HR"
#f'{data_dir}/' + data_type + r'/HR'

data = dict(
    workers_per_gpu=workers_per_gpu,
    train_dataloader=dict(samples_per_gpu=btsize_per_gpu, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=20,
        dataset=dict(
            type=train_dataset_type,
            gt_folder=r'/data/xpz120/SRbenchmark/DIV2K/DIV2K_train_HR',#f'{data_dir}/DIV2K/HR',  
            pipeline=train_pipeline,
            scale=scale_max)),
    val=dict(type=val_dataset_type,
             gt_folder=r'/data/xpz120/SRbenchmark/DIV2K/DIV2K_valid_HR',#gt_path, 
             pipeline=valid_pipeline,
             scale=scale_max),
    test=dict(
        type=test_dataset_type,
        lq_folder=lq_path,
        gt_folder=gt_path, 
        pipeline=test_pipeline,  
        scale=val_scale,
        filename_tmpl='{}') if val_scale <= 4 else 
            dict(type=val_dataset_type, 
                 gt_folder=gt_path, 
                 pipeline=valid_pipeline, 
                 scale=val_scale)
    ) 

# optimizer
optimizers = dict(type='Adam', lr=lr)

# learning policy
iter_per_epoch = 1000
total_iters = 1000 * iter_per_epoch
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[200000, 400000, 600000, 800000],
    gamma=0.5)

checkpoint_config = dict(interval=save_fre, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=val_fre, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
visual_config = None
from torch._C._distributed_c10d import _DEFAULT_PG_TIMEOUT
default_pg_timeout = _DEFAULT_PG_TIMEOUT*20
# runtime settings
dist_params = dict(backend='nccl', timeout=default_pg_timeout)
log_level = 'INFO'
work_dir = f'{run_dir}/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
test_checkpoint_path = f'{run_dir}/{exp_name}/latest.pth' # use --checkpoint None to enable this path in testing
