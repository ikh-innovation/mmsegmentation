from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp


@DATASETS.register_module()
class Pothole600Dataset(CustomDataset):
    CLASSES = ('background', 'pothole')
    PALETTE = [[127, 127, 127], [70, 255, 0]]

    def __init__(self, split, **kwargs):
        super(Pothole600Dataset, self).__init__(img_suffix='.png',
                                                seg_map_suffix='.png',
                                                reduce_zero_label=False,
                                                split=split,
                                                **kwargs)
        assert osp.exists(self.img_dir)
