import torch
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
from helpers import generate_filename

torch.backends.cudnn.enabled = False

CLASS_NAMES = ['unlabeled',
 'ego vehicle',
 'rectification border',
 'out of roi',
 'static',
 'dynamic',
 'ground',
 'road',
 'sidewalk',
 'parking',
 'rail track',
 'building',
 'wall',
 'fence',
 'guard rail',
 'bridge',
 'tunnel',
 'pole',
 'polegroup',
 'traffic light',
 'traffic sign',
 'vegetation',
 'terrain',
 'sky',
 'person',
 'rider',
 'car',
 'truck',
 'bus',
 'caravan',
 'trailer',
 'train',
 'motorcycle',
 'bicycle',
 'license plate']
COLORS = [(0, 0, 0),
 (0, 0, 0),
 (0, 0, 0),
 (0, 0, 0),
 (0, 0, 0),
 (111, 74, 0),
 (81, 0, 81),
 (128, 64, 128),
 (244, 35, 232),
 (250, 170, 160),
 (230, 150, 140),
 (70, 70, 70),
 (102, 102, 156),
 (190, 153, 153),
 (180, 165, 180),
 (150, 100, 100),
 (150, 120, 90),
 (153, 153, 153),
 (153, 153, 153),
 (250, 170, 30),
 (220, 220, 0),
 (107, 142, 35),
 (152, 251, 152),
 (70, 130, 180),
 (220, 20, 60),
 (255, 0, 0),
 (0, 0, 142),
 (0, 0, 70),
 (0, 60, 100),
 (0, 0, 90),
 (0, 0, 110),
 (0, 80, 100),
 (0, 0, 230),
 (119, 11, 32),
 (0, 0, 142)]
INST_ID_TO_ID = {-1: 34,
 0: 0,
 1: 1,
 2: 2,
 3: 3,
 4: 4,
 5: 5,
 6: 6,
 7: 7,
 8: 8,
 9: 9,
 10: 10,
 11: 11,
 12: 12,
 13: 13,
 14: 14,
 15: 15,
 16: 16,
 17: 17,
 18: 18,
 19: 19,
 20: 20,
 21: 21,
 22: 22,
 23: 23,
 24: 24,
 25: 25,
 26: 26,
 27: 27,
 28: 28,
 29: 29,
 30: 30,
 31: 31,
 32: 32,
 33: 33}
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def get_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 35
    cfg.INPUT.MIN_SIZE_TRAIN = 1
    cfg.MODEL.DEVICE = 'cpu'
    # cfg.MODEL.WEIGHTS = APP_ROOT + '/detectron2_fullset_train.pth'
    cfg.MODEL.WEIGHTS = 'http://imake.site/detectron2_fullset_train.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    cityscpaces_metadata = MetadataCatalog.get("cityscpaces").set(thing_classes=CLASS_NAMES,
                                                                  thing_colors=COLORS,
                                                                  thing_dataset_id_to_contiguous_id=INST_ID_TO_ID)
    return predictor, cityscpaces_metadata

def model_predict(predictor, metadata, img_path, file_ext):
    im = cv2.imread(img_path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=0.5,
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    out_filename = os.path.join(os.path.dirname(img_path),
                                generate_filename(file_ext))
    cv2.imwrite(out_filename, out.get_image()[:, :, ::-1])
    return out_filename