_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/cracks_256x256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes=2,
                     norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=2,
                        norm_cfg=norm_cfg),
    #stride fix
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170))
    # test_cfg=dict(mode='whole')
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
)

runner = dict(max_iters=500)
log_config = dict(interval=5,
                  hooks=[
                      dict(type='MMSegWandbHook',
                           init_kwargs={
                               'entity': "iknowhow",
                               'project': "crack-segmentation",
                               'name': 'unet_s5_d16_256x256_cracks'},
                           ),
                      dict(type='TextLoggerHook')
                  ])
evaluation = dict(interval=50, metric='mDice')
# checkpoint_config = dict(interval=150)


log_level = 'ERROR'

# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()



