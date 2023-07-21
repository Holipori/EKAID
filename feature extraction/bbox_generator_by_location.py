
import os
import pickle
import pandas as pd
from detectron2.structures import BoxMode
import argparse
import dataclasses
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from distutils.util import strtobool
from pathlib import Path
from typing import Any, Union, Dict, List
from math import ceil
from numpy import ndarray
import math
import h5py
from os.path import exists
import pydicom
import csv
import yaml

import cv2
import detectron2
import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict
import matplotlib.pyplot as plt
from mytrainer import MyTrainer

from detectron2.config.config import CfgNode as CN
from get_bbox_id import inference
from train_vindr import get_vindr_shape, get_vindr_label2id
from torch.utils.data import Dataset
import gc
from torch.utils.data import DataLoader
setup_logger()


# --- setup ---

def format_pred(labels: ndarray, boxes: ndarray, scores: ndarray) -> str:
    pred_strings = []
    for label, score, bbox in zip(labels, scores, boxes):
        xmin, ymin, xmax, ymax = bbox.astype(np.int64)
        pred_strings.append(f"{label} {score} {xmin} {ymin} {xmax} {ymax}")
    return " ".join(pred_strings)



def predict_batch(predictor: DefaultPredictor, im_list: List[ndarray]) -> List:
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        inputs_list = []
        for original_image in im_list:
            # Apply pre-processing to image.
            if predictor.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            # Do not apply original augmentation, which is resize.
            # image = predictor.aug.get_transform(original_image).apply_image(original_image)
            image = original_image
            try:
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            except:
                image = torch.permute(image, (2,0,1))
            inputs = {"image": image, "height": height, "width": width}
            inputs_list.append(inputs)
        predictions = predictor.model(inputs_list)
    return predictions



def get_vinbigdata_dicts(dataset_dir): # ordered csv
    # json_file = os.path.join(img_dir, "via_region_data.json")
    if exists('data_dicts-png.pkl'):
        with open("data_dicts-png.pkl", "rb") as tf:
            data_dicts = pickle.load(tf)
        return data_dicts
    csv_file = '/home/xinyue/dataset/vinbigdata/train_sorted.csv'
    with open(csv_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        dataset_dicts = []
        image_id_ori = '0'
        i = 0
        for idx, row in enumerate(spamreader):
            if idx == 0:
                continue
            image_id = row[0]
            if image_id != image_id_ori:
                if idx != 1:
                    record["annotations"] = objs
                    dataset_dicts.append(record)

                record = {}
                objs = []

                filename = 'train/' + image_id + '.dicom'
                ds = pydicom.dcmread(os.path.join(dataset_dir, filename))

                height, width = ds.pixel_array.shape
                record["file_name"] = filename
                record["image_id"] = i
                record["height"] = height
                record["width"] = width
                obj = {
                    "bbox": [int(float(row[4])), int(float(row[5])), int(float(row[6])), int(float(row[7]))] if row[4] != '' else [0,0,1,1],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": int(row[2]),
                }
                objs.append(obj)
                i += 1
            else:
                if row[4] != '':
                    obj = {
                        "bbox": [int(float(row[4])), int(float(row[5])), int(float(row[6])), int(float(row[7]))],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": int(row[2]),
                    }
                    objs.append(obj)
            # record["annotations"] = objs
            #
            # dataset_dicts.append(record)
            if idx % 50 == 0:
                print('{} loaded'.format(idx))
            image_id_ori = image_id
    record["annotations"] = objs
    dataset_dicts.append(record)

    with open("data_dicts.pkl", "wb") as tf:
        pickle.dump(dataset_dicts, tf)
        print('dicts saved')
    return dataset_dicts

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

def get_dicts(split):
    data_dicts = get_vinbigdata_dicts('/home/xinyue/dataset/vinbigdata-png')

    l_train_dicts = int(len(data_dicts) * 0.9)
    l_val_dicts = (int(len(data_dicts)) - l_train_dicts)
    # l_test_dicts = int(len(data_dicts)) - l_train_dicts - l_val_dicts
    train_dicts = data_dicts[:l_train_dicts]
    val_dicts = data_dicts[l_train_dicts: l_train_dicts + l_val_dicts]
    # test_dicts = data_dicts[-l_test_dicts:]
    if split == 'train':
        return train_dicts
    elif split == 'val':
        return val_dicts
    elif split == 'test':
        return get_vinbigdata_dicts_test('/home/xinyue/dataset/vinbigdata-png')

def get_mimic_dict(full=True):
    datadir = '/home/xinyue/dataset/mimic-cxr-png/'
    datadict = []
    if full:
        name = '/home/xinyue/VQA_ReGat/data/mimic/mimic_shape_full.pkl'
    else:
        name = '/home/xinyue/VQA_ReGat/data/mimic/mimic_shape.pkl'
    with open(name, 'rb') as f:
        mimic_dataset = pickle.load(f)
    for i,row in enumerate(mimic_dataset):
        record = {}
        filename = datadir + row['image'] + '.png'
        record["file_name"] = filename
        record["image_id"] = row['image']
        record["height"] = 1024
        record["width"] = 1024
        datadict.append(record)

    return datadict

def get_vindr_dict():
    datadir = '/home/xinyue/dataset/vinbigdata-png/'
    datadict = []
    vindr_shape = get_vindr_shape()
    for i,name in enumerate(vindr_shape):
        record = {}
        filename = datadir + name + '.png'
        record["file_name"] = filename
        record["image_id"] = name
        record["height"] = 1024
        record["width"] = 1024
        datadict.append(record)

    return datadict



def save_yaml(filepath: Union[str, Path], content: Any, width: int = 120):
    with open(filepath, "w") as f:
        yaml.dump(content, f, width=width)


def load_yaml(filepath: Union[str, Path]) -> Any:
    with open(filepath, "r") as f:
        content = yaml.full_load(f)
    return content

@dataclass
class Flags:
    # General
    debug: bool = True
    outdir: str = "results/det"

    # Data config
    imgdir_name: str = "vinbigdata-chest-xray-resized-png-256x256"
    split_mode: str = "all_train"  # all_train or valid20
    seed: int = 111
    train_data_type: str = "original"  # original or wbf
    use_class14: bool = False
    # Training config
    iter: int = 10000
    ims_per_batch: int = 2  # images per batch, this corresponds to "total batch size"
    num_workers: int = 0
    lr_scheduler_name: str = "WarmupMultiStepLR"  # WarmupMultiStepLR (default) or WarmupCosineLR
    base_lr: float = 0.00025
    roi_batch_size_per_image: int = 512
    eval_period: int = 10000
    aug_kwargs: Dict = field(default_factory=lambda: {})

    def update(self, param_dict: Dict) -> "Flags":
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self




def save_features(mod, inp, outp):
    feats = inp[0]
    for i in range(batch_size):
        features.append(feats[i*1000: (i+1)*1000])
    predictions.append(outp)
    # feature = inp[0]
    # prediction = outp

def save_features1(mod, inp, outp):
    proposals.append(inp[2])
    # prposal = inp[2]



def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou
def get_center(bb1):
    c1 = [(bb1[2] + bb1[0])/2, (bb1[3] + bb1[1])/2]
    return c1

def get_distance(bb1, bb2):
    c1 = get_center(bb1)
    c2 = get_center(bb2)
    dx = np.abs(c2[0] - c1[0])
    dy = np.abs(c2[1] - c1[1])
    d = np.sqrt(np.square(dx) + np.square(dy))
    return d

def get_angle(coor):
    x1,y1, x2, y2 = coor
    angle = math.atan2(  y2-y1, x2-x1 ) /math.pi *180
    if angle < 0:
        angle += 360
    return angle

def cal_angle(bb1,bb2):
    c1 = get_center(bb1)
    c2 = get_center(bb2)
    return get_angle(c1+c2)

def bbox_relation_type(bb1, bb2, lx=1024, ly=1024):
    if bb1[0]< bb2[0] and bb1[1]< bb2[1] and bb1[2]> bb2[2] and bb1[3]>bb2[3]:
        return 1
    elif bb1[0]> bb2[0] and bb1[1]> bb2[1] and bb1[2]< bb2[2] and bb1[3]<bb2[3]:
        return 2
    elif get_iou(bb1, bb2) >= 0.5:
        return 3
    elif get_distance(bb1, bb2) >= (lx+ly)/3:
        return 0
    angle = cal_angle(bb1,bb2)
    return math.ceil(angle/45) + 3

def reverse_type(type):
    if type == 0:
        return 0
    elif type == 1:
        return 2
    elif type == 2:
        return 1
    elif type == 3:
        return 3
    elif type == 4:
        return 8
    elif type == 5:
        return 9
    elif type == 6:
        return 10
    elif type == 7:
        return 11
    elif type == 8:
        return 4
    elif type == 9:
        return 5
    elif type == 10:
        return 6
    elif type == 11:
        return 7



def get_adj_matrix(bboxes):

    num_pics = len(bboxes)
    n = len(bboxes[0])
    adj_matrix = np.zeros([num_pics,100,100],int)
    for idx in tqdm(range(num_pics)):
        bbs = bboxes[idx]
        for i in range(n):
            for j in range(i,n):
                type = bbox_relation_type(bbs[i],bbs[j])
                adj_matrix[idx,i,j] = type
                adj_matrix[idx,j,i] = reverse_type((type))
    return adj_matrix

def save_h5(dataset, final_features, normalized_bboxes,bboxes, pos_boxes, adj_matrix, test_topk_per_image, pred_classes, times=0, length = 100):
    if dataset == 'mimic':
        filename = './output/mimic_disease_box_by_location/bbox_features_disease_by_location.hdf5'
    # elif dataset == 'vindr':
    #     filename =  './output/vindr_box/bbox_features-25.hdf5'
    if times == 0:
        h5f = h5py.File(filename, 'w')
        image_features_dataset = h5f.create_dataset("image_features", (length, test_topk_per_image, 1024),
                                                    maxshape=(None, test_topk_per_image, 1024),
                                                    chunks=(100, test_topk_per_image, 1024),
                                                    dtype='float32')
        spatial_features_dataset = h5f.create_dataset("spatial_features", (length, test_topk_per_image, 6),
                                                    maxshape=(None, test_topk_per_image, 6),
                                                    chunks=(100, test_topk_per_image, 6),
                                                    dtype='float64')
        image_bb_dataset = h5f.create_dataset("image_bb", (length, test_topk_per_image, 4),
                                                    maxshape=(None, test_topk_per_image, 4),
                                                    chunks=(100, test_topk_per_image, 4),
                                                    dtype='float32')
        pos_boxes_dataset = h5f.create_dataset("pos_boxes", (length, 2),
                                                    maxshape=(None, 2),
                                                    chunks=(100, 2),
                                                    dtype='int64')
        image_adj_matrix_dataset = h5f.create_dataset("image_adj_matrix", (length, 100, 100),
                                                    maxshape=(None, 100, 100),
                                                    chunks=(100, 100, 100),
                                                    dtype='int64')
        bbox_label_dataset = h5f.create_dataset("bbox_label", (length, test_topk_per_image),
                                                    maxshape=(None, test_topk_per_image),
                                                    chunks=(100, test_topk_per_image),
                                                    dtype='int64')
    else:
        h5f = h5py.File(filename, 'a')
        image_features_dataset = h5f['image_features']
        spatial_features_dataset = h5f['spatial_features']
        image_bb_dataset = h5f['image_bb']
        pos_boxes_dataset = h5f['pos_boxes']
        image_adj_matrix_dataset = h5f['image_adj_matrix']
        bbox_label_dataset = h5f['bbox_label']
    # 关键：这里的h5f与dataset并不包含真正的数据，
    # 只是包含了数据的相关信息，不会占据内存空间
    #
    # 仅当使用数组索引操作（eg. dataset[0:10]）
    # 或类方法.value（eg. dataset.value() or dataset.[()]）时数据被读入内存中

    # 调整数据预留存储空间（可以一次性调大些）
    if len(final_features) != length:
        adding = len(final_features)
    else:
        adding = length
    image_features_dataset.resize([times*length+adding, test_topk_per_image, 1024])
    image_features_dataset[times*length:times*length+adding] = final_features

    spatial_features_dataset.resize([times*length+adding, test_topk_per_image, 6])
    spatial_features_dataset[times*length:times*length+adding] = normalized_bboxes

    image_bb_dataset.resize([times*length+adding, test_topk_per_image, 4])
    image_bb_dataset[times*length:times*length+adding] = bboxes

    pos_boxes_dataset.resize([times*length+adding, 2])
    pos_boxes_dataset[times*length:times*length+adding] = pos_boxes

    image_adj_matrix_dataset.resize([times*length+adding, 100, 100])
    image_adj_matrix_dataset[times*length:times*length+adding] = adj_matrix

    bbox_label_dataset.resize([times * length + adding, test_topk_per_image])
    bbox_label_dataset[times * length:times * length + adding] = pred_classes

    h5f.close()

class mydataset(Dataset):
    def __init__(self, dataset):
        super(mydataset, self).__init__()
        if dataset == 'mimic':
            self.datadict = get_mimic_dict()
        elif dataset == 'vindr':
            self.datadict = get_vindr_dict()

    def __getitem__(self, index):
        im = cv2.imread(self.datadict[index]["file_name"])
        return self.datadict[index], im
    def __len__(self):
        return len(self.datadict)

def match_bbx(bb, bb_ana, feat, pred_class, category):
    # find the corresponding anatomical bounding box for each disease bounding box
    bb_disease_dict_anaid = {} # list
    bb_ana_dict_iou = {}
    bb_ana_dict_diseaseid = {}
    for j in range(len(bb_ana)):# initial all iou = 0
        bb_ana_dict_iou[j] = 0
    for i in range(len(bb)): # disease
        for j in range(len(bb_ana)): # ana
            iou = get_iou(bb[i], bb_ana[j])
            if iou > bb_ana_dict_iou[j] and j not in bb_ana_dict_diseaseid: # means this ana_box is not assigned a disease box
                bb_ana_dict_iou[j] = iou
                bb_ana_dict_diseaseid[j] = i
                if i not in bb_disease_dict_anaid:
                    bb_disease_dict_anaid[i] = [j]
                else:
                    bb_disease_dict_anaid[i].append(j)
            elif iou > bb_ana_dict_iou[j] and len(bb_disease_dict_anaid[bb_ana_dict_diseaseid[j]])> 1: # replace the original one
                bb_disease_dict_anaid[bb_ana_dict_diseaseid[j]].remove(j)
                bb_ana_dict_iou[j] = iou
                bb_ana_dict_diseaseid[j] = i
                if i not in bb_disease_dict_anaid:
                    bb_disease_dict_anaid[i] = [j]
                else:
                    bb_disease_dict_anaid[i].append(j)
    out_feat = []
    out_pred_class = []
    for i in range(len(bb)):
        if i in bb_ana_dict_diseaseid:
            id = bb_ana_dict_diseaseid[i]
            out_feat.append(feat[id])
            out_pred_class.append(pred_class[id])
        else:
            out_feat.append(np.zeros(feat.shape[-1]))
            out_pred_class.append(len(category))
    try:
        out_feat = np.stack(out_feat)
    except:
        print('a')
    out_pred_class = np.array(out_pred_class)
    return bb_ana, out_feat, out_pred_class

if __name__ == '__main__':

    dataset = 'mimic'

    category = get_vindr_label2id()
    thing_classes = list(category)
    if dataset == 'mimic':
        imgdir_name = "mimic-cxr-png"
    elif dataset == 'vindr':
        imgdir_name = 'vinbigdata-png'
    flags_dict = {
        "debug": False,
        "outdir": "results/"+dataset,
        "imgdir_name": imgdir_name,
        "split_mode": "all",
        "iter": 100000,
        "roi_batch_size_per_image": 512,
        "eval_period": 1000,
        "lr_scheduler_name": "WarmupCosineLR",
        "base_lr": 0.0001,
        "num_workers": 4,
        "aug_kwargs": {
            "HorizontalFlip": {"p": 0.5},
            "ShiftScaleRotate": {"scale_limit": 0.15, "rotate_limit": 10, "p": 0.5},
            "RandomBrightnessContrast": {"p": 0.5}
        }
    }

    # args = parse()
    print("torch", torch.__version__)
    # flags: Flags = Flags().update(load_yaml(str('results/v9/flags.yaml')))
    flags: Flags = Flags().update(flags_dict)
    print("flags", flags)
    debug = flags.debug
    outdir = Path(flags.outdir)
    os.makedirs(str(outdir), exist_ok=True)
    # flags_dict = dataclasses.asdict(flags)

    # ===============================

    cfg = get_cfg()
    original_output_dir = cfg.OUTPUT_DIR
    cfg.OUTPUT_DIR = str(outdir)
    print(f"cfg.OUTPUT_DIR {original_output_dir} -> {cfg.OUTPUT_DIR}")

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("vinbigdata_train",) # no use. no need to worry about
    cfg.DATASETS.TEST = ()
    # cfg.DATASETS.TEST = ("vinbigdata_train",)
    # cfg.TEST.EVAL_PERIOD = 50
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.num_gpus = 2
    # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = flags.base_lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = flags.iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = flags.roi_batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    ### --- Inference & Evaluation ---
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    # path to the model we just trained
    cfg.MODEL.WEIGHTS = str("checkpoints/model_final_for_vindr.pth")
    print("Original thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)  # 0.05
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0  # set a custom testing threshold
    print("Changed  thresh", cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    predictor = DefaultPredictor(cfg)

    layer_to_hook1 = 'roi_heads'
    # layer_to_hook2 = 'box_head'
    layer_to_hook2 = 'box_predictor'
    layer_to_hook3 = 'fc_relu2'
    for name, layer in predictor.model.named_modules():
        if name == layer_to_hook1:
            layer.register_forward_hook(save_features1)
            for name2, layer2 in layer.named_modules():
                if name2 == layer_to_hook2:
                    # for name3, layer3 in layer2.named_modules():
                    #     if name3 == layer_to_hook3:
                    layer2.register_forward_hook(save_features)

    # hook
    features = []
    proposals = []
    predictions = []

    if dataset == 'mimic':
        DatasetCatalog.register("mimic", get_mimic_dict)
        MetadataCatalog.get("mimic").set(thing_classes=list(category))
        dataset_dicts = get_mimic_dict()
    elif dataset == 'vindr':
        DatasetCatalog.register("vindr", get_vindr_dict)
        MetadataCatalog.get("vindr").set(
            thing_classes=list(category))
        dataset_dicts = get_vindr_dict()


    # for d in ["train", "val", "test"]:
    #     DatasetCatalog.register("vinbigdata_" + d, lambda d=d: get_dicts(d))
    #     MetadataCatalog.get("vinbigdata_" + d).set(thing_classes=['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', 'Lung Opacity','Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis'])



    results_list = []
    index = 0
    out_ids = []
    final_features = []
    bboxes = []
    normalized_bboxes = []
    pos_boxes = []
    pred_classes = []
    adj_matrix = []
    n = 0
    test_topk_per_image = 26 # has to match the ana box number

    batch_size = 1 # only 1 is allowed
    length = 100
    times = 0
    flag = 0

    path = '/home/xinyue/faster-rcnn/output/mimic_ana_box/ana_bbox_features_full.hdf5'
    hf = h5py.File(path,'r')


    resume = False  # remember to check before running
    if resume:
        stopped_batch_num = 17500  # the number you see in the terminal when stooped
        # stopped_batch_num = 75000
        stopped_img_num = stopped_batch_num * 4
        continue_i = (stopped_img_num - length) / 4
        times = int((stopped_img_num - length) / length)
        n = int(continue_i * test_topk_per_image)
    for i in tqdm(range(ceil(len(dataset_dicts) / batch_size))):
        if resume:
            if i < continue_i:
                continue

        bb_ana = hf['image_bb'][i]
        adj_matrix_one = hf['image_adj_matrix'][i]
        adj_matrix.append(adj_matrix_one)

        inds = list(range(batch_size * i, min(batch_size * (i + 1), len(dataset_dicts))))
        dataset_dicts_batch = [dataset_dicts[i] for i in inds]
        im_list = [cv2.imread(d["file_name"]) for d in dataset_dicts_batch]
        outputs_list = predict_batch(predictor, im_list)

        out_ids = (inference(predictions[0], proposals[0], predictor.model.roi_heads.box_predictor.box2box_transform, test_topk_per_image))
        predictions = []
        proposals = []
        for feats, ids, dicts, outputs in zip(features, out_ids, dataset_dicts_batch,outputs_list):
            ids = (ids / len(category)).type(torch.long)
            feat = feats[ids].cpu()
            # f[dicts['image_id']] = feat
            bb = np.array(outputs['instances']._fields['pred_boxes'].tensor.cpu())[:test_topk_per_image]
            pred_class = np.array(outputs['instances']._fields['pred_classes'].cpu())[:test_topk_per_image]
            if len(bb) == 0:
                bb = np.zeros(test_topk_per_image, 4)
                pred_class = np.ones(test_topk_per_image)* len(category)
            bb, feat, pred_class = match_bbx(bb, bb_ana, feat, pred_class, category)
            final_features.append(np.array(feat))
            bboxes.append(bb)
            pred_classes.append(pred_class)
            normalized_bb = np.concatenate((bb/1024,np.zeros((bb.shape[0],2))),1)
            normalized_bboxes.append(normalized_bb)
            pos_boxes.append([n, n+len(normalized_bb)])
            n += len(normalized_bb)
        features = []
        if len(final_features) == length or i == ceil(len(dataset_dicts) / batch_size)-1:
            final_features = np.array(final_features)
            bboxes = np.array(bboxes)
            pred_classes = np.array(pred_classes)
            normalized_bboxes = np.array(normalized_bboxes)
            pos_boxes = np.array(pos_boxes)
            # adj_matrix = get_adj_matrix(bboxes)
            adj_matrix = np.array(adj_matrix)
            save_h5(dataset, final_features, normalized_bboxes,bboxes, pos_boxes, adj_matrix,test_topk_per_image, pred_classes, times= times, length=length)
            final_features = []
            bboxes = []
            normalized_bboxes = []
            pos_boxes = []
            pred_classes = []
            adj_matrix = []
            times += 1
            unreachable_count = gc.collect()
    # pbar.close()

    hf.close()
    print('finished writing')

# this file is generating disease bbx by ana location