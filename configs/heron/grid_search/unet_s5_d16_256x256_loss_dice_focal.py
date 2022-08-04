_base_ = ['../unet_s5_d16_256x256_cracks.py']

model = dict(
    decode_head=dict(
         # ignore_index=0,
         # loss_decode=dict(avg_non_ignore=True)
         loss_decode=[
             # dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.05, 0.95]),
             dict(type='FocalLoss', loss_name='loss_focal', loss_weight=0.7),
             dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=[0.01, 0.99])
         ]
    ),

    auxiliary_head=dict(
        # ignore_index=0,
        # loss_decode=dict(avg_non_ignore=True)
        loss_decode=[
                # dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, class_weight=[0.05, 0.95]),
                dict(type='FocalLoss', loss_name='loss_focal', loss_weight=0.7),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=[0.01, 0.99])
        ]
    )
)

log_config = dict(hooks=[dict(type='MMSegWandbHook', init_kwargs={'name': 'unet_s5_d16_256x256_loss_dice_focal'})])
