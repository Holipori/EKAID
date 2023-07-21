# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
import time
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import csv

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
import pydicom
import pickle


# im = cv2.imread("./input.jpg")
# plt.imshow(im[:,:,[2,1,0]])
# plt.show()
#
# cfg = get_cfg()
# # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
#
# # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
#
# # We can use `Visualizer` to draw the predictions on the image.
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# plt.imshow(out.get_image())
# plt.show()



def get_vinbigdata_dicts_test(dataset_dir):
    # json_file = os.path.join(img_dir, "via_region_data.json")
    if os.path.exists('data_dicts-png_test.pkl'):
        with open("data_dicts-png_test.pkl", "rb") as tf:
            data_dicts = pickle.load(tf)
        return data_dicts
    dataset_dir = os.path.join(dataset_dir,'test')
    files = os.listdir(dataset_dir)
    dataset_dicts = []
    for idx, file in enumerate(files):
        record = {}
        filename = os.path.join(dataset_dir,file)
        ds = pydicom.dcmread('/home/xinyue/dataset/vinbigdata/test/'+file.replace('.png','.dicom'))
        height, width = ds.pixel_array.shape
        record["file_name"] = filename
        record["image_id"] = file.replace('.png',"")
        record["height"] = height
        record["width"] = width
        dataset_dicts.append(record)

        if idx % 50 == 0:
            print('{} loaded'.format(idx))
    with open("data_dicts-png_test.pkl", "wb") as tf:
        pickle.dump(dataset_dicts, tf)
    return dataset_dicts


data_dicts = get_vinbigdata_dicts_test('/home/xinyue/dataset/vinbigdata-png')

l_train_dicts = int(len(data_dicts) * 0.8)
l_val_dicts = int((int(len(data_dicts)) - l_train_dicts))
train_dicts = data_dicts[:l_train_dicts]
val_dicts = data_dicts[l_train_dicts: l_train_dicts + l_val_dicts]
