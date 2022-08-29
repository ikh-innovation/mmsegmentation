_base_ = ['../segformer_mit-b0_cracks.py']
log_config = dict(hooks=[dict(type='MMSegWandbHook', init_kwargs={'entity': "iknowhow",
                                                                  'project': "crack-segmentation",
                                                                  'name': 'segformer_mit-b0_focal_cracks'})])

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='FocalLoss', loss_name='loss_focal', loss_weight=1.),
        ]
    )
)
