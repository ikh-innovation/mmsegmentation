_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/pothole600.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_100epochs.py'
]

log_config = dict(hooks=[dict(type='MMSegWandbHook', init_kwargs={'entity': "iknowhow",
                                                                  'project': "pothole600",
                                                                  'name': 'segformer_mit-b5_dice_OHEM_pothole600'})])

data = dict(
    samples_per_gpu=2, workers_per_gpu=2
)

# model settings
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 6, 40, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=2,
        loss_decode=[
            # dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.05, 0.95]),
            # dict(type='FocalLoss', loss_name='loss_focal', loss_weight=0.7),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=[0.2, 0.8]),
        ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
    ))
