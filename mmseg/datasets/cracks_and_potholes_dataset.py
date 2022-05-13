from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp


@DATASETS.register_module()
class CracksAndPotholesDataset(CustomDataset):

  CLASSES = ('normal', 'crack')
  PALETTE = [[255,0,0], [0, 255, 0]]

  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='_RAW.jpg', seg_map_suffix='_CRACK.png',
                     split=split, **kwargs)
    assert osp.exists(self.img_dir)
    assert self.split is not None
    