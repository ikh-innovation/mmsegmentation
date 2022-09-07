_base_ = ['../segformer_mit-b0_640x640_cracks_and_potholes.py']
log_config = dict(hooks=[dict(type='MMSegWandbHook', init_kwargs={'entity': "iknowhow",
                                                                  'project': "crack-segmentation",
                                                                  'name': 'segformer_mit-b0_dice_ignore_640x640_cracks_and_potholes'})])

model = dict(
    decode_head=dict(
        num_classes=3,
        loss_decode=[dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=[0.001, 0.999, 0.999])],
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
    )
)


data = dict(samples_per_gpu=4, workers_per_gpu=2)