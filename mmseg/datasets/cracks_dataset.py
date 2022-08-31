from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp


@DATASETS.register_module()
class CracksDataset(CustomDataset):

  CLASSES = ('background', 'crack')
  PALETTE = [[127,127,127], [255, 70, 0]]

  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='_RAW.jpg', seg_map_suffix='_LABELS.png',
                     split=split, **kwargs)
    assert osp.exists(self.img_dir)
    assert self.split is not None
