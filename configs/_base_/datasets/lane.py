# dataset settings
dataset_type = 'LaneDataset'
data_root = 'lane/'
data = 'Dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (1024, 640)
crop_size = (640, 640)
stride = (128, 128)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=255, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        img_ratios=[1.],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    # samples_per_gpu=2,
    # workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=data,
        ann_dir=data,
        pipeline=train_pipeline,
        split='splits/train.txt'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=data,
        ann_dir=data,
        pipeline=test_pipeline,
        split='splits/val1.txt'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=data,
        ann_dir=data,
        pipeline=test_pipeline,
        split='splits/val1.txt'))
