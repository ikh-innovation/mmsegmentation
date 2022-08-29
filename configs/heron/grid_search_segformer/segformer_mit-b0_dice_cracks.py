_base_ = ['../segformer_mit-b0_cracks.py']
log_config = dict(hooks=[dict(type='MMSegWandbHook', init_kwargs={'entity': "iknowhow",
                                                                  'project': "crack-segmentation",
                                                                  'name': 'segformer_mit-b0_dice_cracks'})])

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=[0.001, 0.999])
        ]
    )
)
