_base_ = ['../segformer_mit-b0_256x256_cracks.py']
log_config = dict(hooks=[dict(type='MMSegWandbHook', init_kwargs={'entity': "iknowhow",
                                                                  'project': "crack-segmentation",
                                                                  'name': 'segformer_mit-b0_dice_pad_val_cracks'})])

crop_size = (640, 640)
train_pipeline = [dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255)]

data = dict(train=dict(pipeline=train_pipeline))
