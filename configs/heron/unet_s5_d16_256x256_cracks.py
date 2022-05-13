_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/cracks_and_potholes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes=2,
                     norm_cfg=norm_cfg,
                     # ignore_index=0,
                     # loss_decode=dict(avg_non_ignore=True)
                     loss_decode=[
                         dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.05, 0.95]),
                         dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0, class_weight=[0.05, 0.95])
                     ]
                     ),

    auxiliary_head=dict(num_classes=2,
                        norm_cfg=norm_cfg,
                        # ignore_index=0,
                        # loss_decode=dict(avg_non_ignore=True)
                        loss_decode=[
                                dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.05, 0.95]),
                                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0, class_weight=[0.05, 0.95])
                            ]
                        ),
    #stride fix
    test_cfg=dict(crop_size=(256, 256), stride=(170, 170))
)

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=1,
)


runner = dict(max_iters=200)
log_config = dict(interval=5,
                  hooks=[
                      dict(type='TensorboardLoggerHook'),
                      dict(type='TextLoggerHook')
                  ])
evaluation = dict(interval=50, metric='mDice')
checkpoint_config = dict(interval=200)


# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()


