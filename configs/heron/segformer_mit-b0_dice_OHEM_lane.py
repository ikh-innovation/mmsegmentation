_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/lane.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_100epochs.py'
]

log_config = dict(hooks=[dict(type='MMSegWandbHook', init_kwargs={'entity': "iknowhow",
                                                                  'project': "lane",
                                                                  'name': 'segformer_mit-b0_dice_OHEM_lane'})])


checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
data = dict(
    samples_per_gpu=2, workers_per_gpu=2
)
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
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0),
        ],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
    )
)
