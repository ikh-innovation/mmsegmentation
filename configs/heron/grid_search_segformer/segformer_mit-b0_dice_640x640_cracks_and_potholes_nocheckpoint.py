_base_ = ['../segformer_mit-b0_640x640_cracks_and_potholes.py']
log_config = dict(hooks=[dict(type='MMSegWandbHook', init_kwargs={'entity': "iknowhow",
                                                                  'project': "crack-segmentation",
                                                                  'name': 'segformer_mit-b0_dice_640x640_cracks_and_potholes_nocheckpoint'})])


model = dict(pretrained=None)

data = dict(samples_per_gpu=4, workers_per_gpu=2)