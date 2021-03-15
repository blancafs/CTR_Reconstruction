import abc

from ctr.common.common import CtrClass


class SegmentationStrategyAbstract(CtrClass):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def segmentImg(self, img):
        ''' required method '''

    @abc.abstractmethod
    def segmentImgs(self, list, img):
        ''' required method '''


class Segmentor(CtrClass):
    def __init__(self, strategy):
        self.segmentStrategy = strategy

    def segmentImg(self, img):
        self.segmentStrategy.segmentImg(img)

    def segmentImgs(self, imgs, *args):
        self.segmentStrategy.segmentImgs(imgs, *args)
