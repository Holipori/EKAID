# this file is used for vindr dataset training

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
from typing import Any, Union

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
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Dict
import matplotlib.pyplot as plt
from mytrainer import MyTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from evaluator import VinbigdataEvaluator
from detectron2.data import build_detection_test_loader

from detectron2.config.config import CfgNode as CN
setup_logger()


# --- setup ---


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_vindr_label2id(filter_class = None):
    if filter_class:
        return transform_filtered_class_to_dict(filter_class)
    dict = {}
    dict['Aortic enlargement'] = 0
    dict['Atelectasis'] = 1
    dict['Cardiomegaly'] = 2
    dict['Calcification'] = 3
    dict['Clavicle fracture'] = 4
    dict['Consolidation'] = 5
    dict['Edema'] = 6
    dict['Emphysema'] = 7
    dict['Enlarged PA'] = 8
    dict['ILD'] = 9
    dict['Infiltration'] = 10
    dict['Lung cavity'] = 11
    dict['Lung cyst'] = 12
    dict['Lung Opacity'] = 13
    dict['Mediastinal shift'] = 14
    dict['Nodule/Mass'] = 15
    dict['Pulmonary fibrosis'] = 16
    dict['Pneumothorax'] = 17
    dict['Pleural thickening'] = 18
    dict['Pleural effusion'] = 19
    dict['Rib fracture'] = 20
    dict['Other lesion'] = 21
    # dict['No finding'] = 22
    return dict

def transform_filtered_class_to_dict(classes):
    dict = {}
    for item in classes:
        dict[item] = len(dict)
    return dict

def get_vindr_dicts(vindr_shape, split, filter_class, filter_empty): # ordered csv
    # json_file = os.path.join(img_dir, "via_region_data.json")
    if exists('data_dicts-vindr.pkl'):
        with open("data_dicts-vindr.pkl", "rb") as tf:
            data_dicts = pickle.load(tf)
        return data_dicts

    label2id = get_vindr_label2id(filter_class)

    csv_file = '/home/xinyue/dataset/physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_'+ split +'.csv'

    with open(csv_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        reader = csv.DictReader(csvfile)
        dataset_dicts = []
        image_id_ori = '0' # initialization
        i = 0
        for idx, row in enumerate(reader):
            image_id = row['image_id']
            if image_id != image_id_ori:
                if idx != 0:
                    record["annotations"] = objs
                    if filter_empty:
                        if objs != []:
                            dataset_dicts.append(record)
                    else:
                        dataset_dicts.append(record)

                record = {}
                objs = []

                filename = 'train/' + image_id + '.dicom'
                # ds = pydicom.dcmread(os.path.join(dataset_dir, filename))
                filename = '/home/xinyue/dataset/vinbigdata-png/' + image_id+'.png'
                height, width = vindr_shape[image_id]
                record["file_name"] = filename
                record["image_id"] = i
                record["height"] = 1024
                record["width"] = 1024



                if row['x_min'] != '':
                    if filter_class:
                        if row['class_name'] not in label2id:
                            image_id_ori = image_id
                            continue
                    x1 = max(float(row['x_min']) * (1024/width), 0)
                    y1 = max(float(row['y_min']) * (1024/height), 0)
                    x2 = min(float(row['x_max']) * (1024/width),1024)
                    y2 = min(float(row['y_max']) * (1024/height),1024)
                    if x1>x2 or y1 >y2:
                        i += 1
                        image_id_ori = image_id
                        continue


                    obj = {
                        "bbox": [x1,y1,x2,y2] if row['x_min'] != '' else '',
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": int(label2id[row['class_name']]),
                    }
                    objs.append(obj)
                i += 1
            else:
                if row['x_min'] != '':
                    if filter_class:
                        if row['class_name'] not in label2id:
                            image_id_ori = image_id
                            continue
                    x1 = max(float(row['x_min']) * (1024/width), 0)
                    y1 = max(float(row['y_min']) * (1024/height), 0)
                    x2 = min(float(row['x_max']) * (1024/width),1024)
                    y2 = min(float(row['y_max']) * (1024/height),1024)
                    if x1 > x2 or y1 > y2:
                        image_id_ori = image_id
                        continue
                    obj = {
                        "bbox": [x1,y1,x2,y2],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": int(label2id[row['class_name']]),
                    }
                    objs.append(obj)
            # record["annotations"] = objs
            #
            # dataset_dicts.append(record)
            if i % 50 == 0:
                print('{} loaded'.format(idx))
            image_id_ori = image_id
    record["annotations"] = objs
    if filter_empty:
        if objs != []:
            dataset_dicts.append(record)
    else:
        dataset_dicts.append(record)

    # with open("data_dicts-vindr", "wb") as tf:
    #     pickle.dump(dataset_dicts, tf)
    #     print('dicts saved')
    return dataset_dicts

def get_vindr_shape():
    with open("vindr-shape.pkl", "rb") as tf:
        vindr_shape = pickle.load(tf)
        return vindr_shape
    path = '/home/xinyue/dataset/physionet.org/files/vindr-cxr/1.0.0/'
    split = ['train','test']
    vindr_shape = {}
    for sp in split:
        dicom_dir = os.path.join(path,sp)
        file_names = os.listdir(dicom_dir)
        for i in tqdm(range(len(file_names))):
            file_name = file_names[i]
            if file_name == 'index.html':
                continue
            file = os.path.join(dicom_dir, file_name)
            png_path = os.path.join('/home/xinyue/dataset/vinbigdata-png', file_name.replace('.dicom','.png'))

            ds = pydicom.dcmread(file)
            height, width = ds.pixel_array.shape
            name = file_name.replace('.dicom','')
            vindr_shape[name] = (height, width)
    with open("vindr-shape.pkl", "wb") as tf:
        pickle.dump(vindr_shape,tf)
    return vindr_shape

def get_dicts(split, filter_class, filter_empty):
    vindr_shape = get_vindr_shape()
    data_dicts = get_vindr_dicts(vindr_shape, split, filter_class, filter_empty)
    return data_dicts

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
    num_workers: int = 4
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


def run(filter_class = None, filter_empty= False):
    category = get_vindr_label2id(filter_class)
    category_name_to_id = category
    vindr_shape = get_vindr_shape()

    flags_dict = {
        "debug": False,
        "imgdir_name": "vinbigdata-png",
        "split_mode": "valid20",
        "iter": 200000,
        "roi_batch_size_per_image": 512,
        "eval_period": 5000,
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
    flags = Flags().update(flags_dict)
    if filter_class:
        flags_dict = {"outdir": "results/vindr-online-filter" }
    else:
        flags_dict = {"outdir": "results/vindr-online"}
    flags = flags.update(flags_dict)
    print("flags", flags)
    debug = flags.debug
    outdir = Path(flags.outdir)
    os.makedirs(str(outdir), exist_ok=True)
    flags_dict = dataclasses.asdict(flags)
    save_yaml(outdir / "flags.yaml", flags_dict)

    # --- Read data ---
    # inputdir = Path("/home/xinyue/dataset/")
    # datadir = inputdir / "vinbigdata"
    # imgdir = inputdir / flags.imgdir_name
    #
    # # Read in the data CSV files
    # train_df = pd.read_csv(datadir / "train.csv")
    # train = train_df  # alias
    # sample_submission = pd.read_csv(datadir / 'sample_submission.csv')

    #--------------------------------------
    train_data_type = flags.train_data_type
    # if flags.use_class14:
    #     thing_classes.append("No finding")

    split_mode = flags.split_mode
    if split_mode == "all_train":
        DatasetCatalog.register(
            "vindr_train",
            lambda: get_dicts(
                'train',filter_class, filter_empty
            ),
        )
        MetadataCatalog.get("vindr_train").set(thing_classes=list(category))
    elif split_mode == "valid20":
        # To get number of data...
        # n_dataset = len(
        #     get_vindr_dicts(vindr_shape,'train')
        # )
        # n_train = int(n_dataset)
        # print("n_dataset", n_dataset, "n_train", n_train)
        # rs = np.random.RandomState(flags.seed)
        # inds = rs.permutation(n_dataset)
        # train_inds, valid_inds = inds[:n_train], inds[n_train:]
        DatasetCatalog.register(
            "vindr_train",
            lambda: get_dicts('train',filter_class, filter_empty),
        )
        MetadataCatalog.get("vindr_train").set(thing_classes=list(category))
        DatasetCatalog.register(
            "vindr_test",
            lambda: get_dicts('test',filter_class, filter_empty),
        )
        MetadataCatalog.get("vindr_test").set(thing_classes=list(category))
    else:
        raise ValueError(f"[ERROR] Unexpected value split_mode={split_mode}")


    dataset_dicts = get_dicts('train',filter_class, filter_empty)
    # Visualize data...
    # anomaly_image_ids = train.query("class_id != 14")["image_id"].unique()
    # # train_meta = pd.read_csv(imgdir/"train_meta.csv")
    # train_meta = pd.read_csv(datadir/"train.csv")
    # anomaly_inds = np.argwhere(train_meta["image_id"].isin(anomaly_image_ids).values)[:, 0]

    vinbigdata_metadata = MetadataCatalog.get("vindr_train")

    # cols = 3
    # rows = 3
    # fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
    # axes = axes.flatten()
    #
    # for index in range(cols * rows):
    #     while 1:
    #         d = random.sample(dataset_dicts, 1)
    #         if d[0]['annotations'] != []:
    #             d = d[0]
    #             break
    #
    #     ax = axes[index]
    #     # print(anom_ind)
    #     # d = dataset_dicts[anom_ind]
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=vinbigdata_metadata, scale=0.5)
    #     out = visualizer.draw_dataset_dict(d)
    #     # cv2_imshow(out.get_image()[:, :, ::-1])
    #     #cv2.imwrite(str(outdir / f"vinbigdata{index}.jpg"), out.get_image()[:, :, ::-1])
    #     ax.imshow(out.get_image()[:, :, ::-1])
    #     ax.set_title(f"image_id {anomaly_image_ids[index]}")
    # plt.show()

    #===================================
    cfg = get_cfg()
    cfg.aug_kwargs = CN(flags.aug_kwargs)  # pass aug_kwargs to cfg

    original_output_dir = cfg.OUTPUT_DIR
    cfg.OUTPUT_DIR = str(outdir)
    print(f"cfg.OUTPUT_DIR {original_output_dir} -> {cfg.OUTPUT_DIR}")

    config_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    cfg.DATASETS.TRAIN = ("vindr_train",)
    if split_mode == "all_train":
        cfg.DATASETS.TEST = ()
    else:
        cfg.DATASETS.TEST = ("vindr_test",)
        cfg.TEST.EVAL_PERIOD = flags.eval_period

    cfg.DATALOADER.NUM_WORKERS = flags.num_workers
    # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)
    # cfg.MODEL.WEIGHTS = os.path.join('results/vindr-online', "model_final.pth")
    cfg.SOLVER.IMS_PER_BATCH = flags.ims_per_batch
    cfg.SOLVER.LR_SCHEDULER_NAME = flags.lr_scheduler_name
    cfg.SOLVER.BASE_LR = flags.base_lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = flags.iter
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000  # Small value=Frequent save need a lot of storage.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = flags.roi_batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(category)
    # NOTE: this config means the number of classes,
    # but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=True # include empty annotation images

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    # trainer.train()


    # evaluate
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    if filter_class:
        cfg.MODEL.WEIGHTS = os.path.join('results/vindr-online-filter', "model_final.pth")
        evaluator = VinbigdataEvaluator("vindr_test", output_dir="./results/vindr-online-filter")
    else:
        cfg.MODEL.WEIGHTS = os.path.join('results/vindr-online', "model_final.pth")  # path to the model we just trained
        evaluator = VinbigdataEvaluator("vindr_test", output_dir="./results/vindr-online")
    # evaluator = COCOEvaluator("vindr_test", output_dir="./results/vindr")
    val_loader = build_detection_test_loader(cfg, "vindr_test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == '__main__':
    filter_empty = False

    filter_class = None
    # filter_class = ['Aortic enlargement', 'Cardiomegaly', 'Pulmonary fibrosis', 'Pleural thickening', 'Pleural effusion', 'Lung Opacity', 'Nodule/Mass', 'Other lesion']
    run(filter_class, filter_empty)