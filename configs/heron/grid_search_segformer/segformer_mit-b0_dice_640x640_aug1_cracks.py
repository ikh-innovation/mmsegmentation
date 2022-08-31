_base_ = ['../segformer_mit-b0_256x256_cracks.py']
log_config = dict(hooks=[dict(type='MMSegWandbHook', init_kwargs={'entity': "iknowhow",
                                                                  'project': "crack-segmentation",
                                                                  'name': 'segformer_mit-b0_dice_640x640_aug1_cracks'})])

img_norm_cfg = dict(
    mean=[125.45, 123.62, 121.77], std=[28.85, 25.91, 25.39], to_rgb=True)
img_scale = (1024, 640)
crop_size = (640, 640)
stride = (128, 128)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=255, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(train=dict(pipeline=train_pipeline),
            samples_per_gpu=8, workers_per_gpu=2)
