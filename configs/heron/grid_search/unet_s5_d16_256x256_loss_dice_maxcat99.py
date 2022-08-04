_base_ = ['../unet_s5_d16_256x256_cracks.py']

model = dict(
    decode_head=dict(
         # ignore_index=0,
         # loss_decode=dict(avg_non_ignore=True)
         loss_decode=[
             dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=[0.01, 0.99])
         ]
    ),

    auxiliary_head=dict(
        # ignore_index=0,
        # loss_decode=dict(avg_non_ignore=True)
        loss_decode=[
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=[0.01, 0.99])
        ]
    )
)

data = dict(
    train=dict(
        pipeline=[
            dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.999)
        ]
    )
)

log_config = dict(hooks=[dict(type='MMSegWandbHook', init_kwargs={'name': 'unet_s5_d16_256x256_loss_dice_maxcat99'})])