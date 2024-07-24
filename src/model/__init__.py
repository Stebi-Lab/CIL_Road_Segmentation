from src.model.seg_former import SegFormerImpl
from src.model.simple_unet import UnetModel
from src.model.nothing_model import TestModel
from src.model.unet_plusplus import UNetPlusPlusModel
from src.model.unet_plusplus_pretrained import UNetPlusPlusModel_Pretrained

modelMappingDict = {"TestModel": TestModel,"UnetModel": UnetModel, "SegFormer": SegFormerImpl, "UnetPlusPlusModel": UNetPlusPlusModel, "UnetPlusPlusModel_Pretrained": UNetPlusPlusModel_Pretrained}