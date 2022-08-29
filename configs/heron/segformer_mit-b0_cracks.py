_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/cracks_and_potholes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_100epochs.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

# model = dict(pretrained=checkpoint, decode_head=dict(num_classes=2))

model = dict(
    pretrained=checkpoint,
    decode_head=dict(
        num_classes=2,
        # ignore_index=None,
        # ignore_index=0,
        # loss_decode=dict(avg_non_ignore=True)
        loss_decode=[
            # dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.05, 0.95]),
            # dict(type='FocalLoss', loss_name='loss_focal', loss_weight=0.7),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=[0.001, 0.999])
        ]
    )
)

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.000003,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=32, workers_per_gpu=8)

log_config = dict(interval=5,
                  hooks=[
                      dict(type='TextLoggerHook'),
                      dict(type='MMSegWandbHook',
                           init_kwargs={
                               'entity': "iknowhow",
                               'project': "crack-segmentation",
                               'name': 'segformer_mit-b0_cracks'},
                           interval=100,
                           num_eval_images=10,
                           log_checkpoint=True,
                           log_checkpoint_metadata=True,
                           )
                  ])
# evaluation = dict(interval=100, metric=['mIoU', 'mDice', 'mFscore'], pre_eval=True)
# # evaluation = dict(interval=100, metric='mIoU')
# checkpoint_config = dict(interval=100)
# runner = dict(type='IterBasedRunner', max_iters=200)
#
workflow = [('train', 10), ('val', 1)]

# log_level = 'ERROR'

# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()