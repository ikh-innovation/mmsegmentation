_base_ = ['../segformer_mit-b0_640x640_cracks_and_potholes.py']
log_config = dict(hooks=[dict(type='MMSegWandbHook', init_kwargs={'entity': "iknowhow",
                                                                  'project': "crack-segmentation",
                                                                  'name': 'segformer_mit-b0_dice_ignore_640x640_cracks_and_potholes'})])

model = dict(
    decode_head=dict(
        num_classes=2,
        # ignore_index=None,
        ignore_index=0,
        loss_decode=[dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, avg_non_ignore=True)]
    )
)
