# Some basic setup:
# Setup detectron2 logger
import detectron2
import torch
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
from os.path import exists
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from evaluator import VinbigdataEvaluator
from detectron2.data import build_detection_test_loader
from tqdm import tqdm
import torch

# CUDA_LAUNCH_BLOCKING=1



def get_vindr_label2id():
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


def get_vindr_dicts(vindr_shape, split): # ordered csv
    # json_file = os.path.join(img_dir, "via_region_data.json")
    if exists('data_dicts-vindr.pkl'):
        with open("data_dicts-vindr.pkl", "rb") as tf:
            data_dicts = pickle.load(tf)
        return data_dicts

    label2id = get_vindr_label2id()

    csv_file = '/home/xinyue/dataset/physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_'+ split +'.csv'

    with open(csv_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        reader = csv.DictReader(csvfile)
        dataset_dicts = []
        image_id_ori = '0'
        i = 0
        for idx, row in enumerate(reader):
            image_id = row['image_id']
            if image_id != image_id_ori:
                if idx != 0:
                    record["annotations"] = objs
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
                    x1 = float(row['x_min']) * (1024/width)
                    y1 = float(row['y_min']) * (1024/height)
                    x2 = float(row['x_max']) * (1024/width)
                    y2 = float(row['y_max']) * (1024/height)
                    if x1>x2 or y1 >y2:
                        i += 1
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
                    x1 = float(row['x_min']) * (1024 / width)
                    y1 = float(row['y_min']) * (1024 / height)
                    x2 = float(row['x_max']) * (1024 / width)
                    y2 = float(row['y_max']) * (1024 / height)
                    if x1>x2 or y1 >y2:
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
    dataset_dicts.append(record)

    # with open("data_dicts-vindr", "wb") as tf:
    #     pickle.dump(dataset_dicts, tf)
    #     print('dicts saved')
    return dataset_dicts

def convert_shape(mimic_shape):
    shape ={}
    print('start converting shape')
    for item in mimic_shape:
        shape[item['image']] = (item['height'], item['width'])
    print('finish converting shape')
    return shape

def get_Ratio(old_size):
     # old_size is in (height, width) format
    width = 224
    ratio = float(width) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    # im = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)

    delta_w = width - new_size[1]
    delta_h = width - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)


    return top, bottom, left, right, ratio


def get_Original_Coordinates(coordinates, scales):
    top, bottom, left, right, ratio = scales
    x1, x2, y1, y2 = coordinates

    # Map coordinates to original image
    scale = 1 / ratio
    original_x1 = int(scale * (x1 - left))
    original_x2 = int(scale * (x2 - left))
    original_y1 = int(scale * (y1 - top))
    original_y2 = int(scale * (y2 - top))
    return original_x1, original_x2, original_y1, original_y2

def get_vindr_shape():
    with open("/home/xinyue/faster-rcnn/vindr-shape.pkl", "rb") as tf:
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
    with open("/home/xinyue/faster-rcnn/vindr-shape.pkl", "wb") as tf:
        pickle.dump(vindr_shape,tf)
    return vindr_shape


def get_dicts(split):
    vindr_shape = get_vindr_shape()
    data_dicts = get_vindr_dicts(vindr_shape, split)
    # data_dicts = get_mimic_ana_dicts('/home/xinyue/dataset/chest-imagenome/1.0.0/silver_dataset/scene_graph/')

    return data_dicts
def get_kg():
    kg_dict = {}
    # anatomical part
    kg_dict['Right lung'] = 'Lung'
    kg_dict['Right upper lung zone'] = 'Lung'
    kg_dict['Right mid lung zone'] = 'Lung'
    kg_dict['Right lower lung zone'] = 'Lung'
    kg_dict['Hilar Area of the Right Lung'] = 'Lung'
    kg_dict['Apical zone of right lung'] = 'Lung'
    kg_dict['right costophrenic sulcus;;Right costodiaphragmatic recess'] = 'Pleural'
    kg_dict['right cardiophrenic sulcus'] = 'Pleural'
    kg_dict['Right hemidiaphragm'] = 'Pleural' # probably
    kg_dict['Left lung'] = 'Lung'
    kg_dict['Left upper lung zone'] = 'Lung'
    kg_dict['Left mid lung zone'] = 'Lung'
    kg_dict['Left lower lung zone'] = 'Lung'
    kg_dict['Hilar Area of the Left Lung'] = 'Lung'
    kg_dict['Apical zone of left lung'] = 'Lung'
    kg_dict['left costophrenic sulcus;;Left costodiaphragmatic recess'] = 'Pleural'
    kg_dict['Left hemidiaphragm'] = 'Pleural' # probably

    kg_dict['Trachea&&Main Bronchus'] = 'Lung'
    kg_dict['Vertebral column'] = 'Spine'
    kg_dict['Right clavicle'] = 'Bone'
    kg_dict['Left clavicle'] = 'Bone'
    kg_dict['Aortic arch structure'] = 'Heart'
    kg_dict['Mediastinum'] = 'Mediastinum'
    kg_dict['Superior mediastinum'] = 'Mediastinum'
    kg_dict['Superior vena cava structure'] = 'Heart'
    kg_dict['Cardiac shadow viewed radiologically;;Heart'] = 'Heart'
    kg_dict['Structure of left margin of heart'] = 'Heart'
    kg_dict['Right border of heart viewed radiologically'] = 'Heart'
    kg_dict['cavoatrial'] = 'Heart'
    kg_dict['Right atrial structure'] = 'Heart'
    kg_dict['Descending aorta'] = 'Heart'
    kg_dict['Structure of carina'] = 'Lung'

    kg_dict['Structure of left upper quadrant of abdomen'] = 'Abdomen' # new group
    kg_dict['Structure of right upper quadrant of abdomen'] = 'Abdomen'# new group
    kg_dict['Abdominal cavity'] = 'Abdomen'# new group
    kg_dict['left cardiophrenic sulcus'] = 'Pleural'

    return kg_dict
def filter_classes(outputs, category, name = None):
    if name == None:
        return outputs
    target_labels = set()

    #filter by organ
    # kg_dict = get_kg()
    # target_names = []
    # for item in kg_dict:
    #     if kg_dict[item] == name:
    #         target_names.append(item)

    # filter by label
    target_names = [name]

    print(target_names)
    # if name == 'Pleural':
    #     target_names = ['right costophrenic sulcus;;Right costodiaphragmatic recess',
    #                     'right cardiophrenic sulcus',
    #                     'Right hemidiaphragm',
    #                     'left costophrenic sulcus;;Left costodiaphragmatic recess',
    #                     'Left hemidiaphragm',
    #                     ]
    for item in category:
        if item in target_names:
            target_labels.add(category[item])
    target_index = []
    if 'annotations' not in outputs:
        for i, item in enumerate(outputs['instances']._fields['pred_classes']):
            if int(item) in target_labels:
                target_index.append(i)
        target_index = np.array(target_index)
        outputs['instances']._fields['pred_boxes'] = outputs['instances']._fields['pred_boxes'][target_index]
        outputs['instances']._fields['scores'] = outputs['instances']._fields['scores'][target_index]
        outputs['instances']._fields['pred_classes'] = outputs['instances']._fields['pred_classes'][target_index]
        return outputs
    else:
        out = []
        for i, item in enumerate(outputs['annotations']):
            if item['category_id'] in target_labels:
                out.append(item)
        # target_index = np.array(target_index)
        outputs['annotations'] = out
        return outputs

def my_predictor(threshold = 0.2):
    category = get_vindr_label2id()

    for d in ["train", "test"]:
        DatasetCatalog.register("vindr_" + d, lambda d=d: get_dicts(d))
        MetadataCatalog.get("vindr_" + d).set(thing_classes=list(category))
    vindr_metadata = MetadataCatalog.get("vindr_train")

    dataset_dicts = get_dicts('train')
    # for d in random.sample(dataset_dicts, 1):
    #     # for d in dataset_dicts[:20]:
    #     print(d["file_name"])
    #
    #     if d['annotations'] == []:
    #         continue
    #
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=vindr_metadata, scale=1)
    #     out = visualizer.draw_dataset_dict(d)
    #
    #     plt.figure(dpi=300)
    #     plt.imshow(out.get_image())
    #     plt.show()

    cfg = get_cfg()
    cfg.OUTPUT_DIR = 'results/vindr'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("vindr_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-1-3000.pth")
    # cfg.MODEL.WEIGHTS = os.path.join('results/v9', "model_final.pth")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
                                     "model_0069999.pth")  # this is the path of the trained vinbigdata model
    cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 200000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(
        category)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False  # include empty annotation images

    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # print('start training')
    # # trainer.train()
    # print('finish training')

    # testing plot
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join('/home/xinyue/faster-rcnn','results/vindr-online', "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    vindr_metadata = MetadataCatalog.get("vindr_test")
    return predictor, vindr_metadata


def run():
    category = get_vindr_label2id()


    for d in ["train", "test"]:
        DatasetCatalog.register("vindr_" + d, lambda d=d: get_dicts(d))
        MetadataCatalog.get("vindr_" + d).set(thing_classes=list(category))
    vindr_metadata = MetadataCatalog.get("vindr_train")


    dataset_dicts = get_dicts('train')
    for d in random.sample(dataset_dicts, 1):
    # for d in dataset_dicts[:20]:
        print(d["file_name"])

        if d['annotations'] == []:
            continue

        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=vindr_metadata, scale=1)
        out = visualizer.draw_dataset_dict(d)

        plt.figure(dpi=300)
        plt.imshow(out.get_image())
        plt.show()

    cfg = get_cfg()
    cfg.OUTPUT_DIR = 'results/vindr'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("vindr_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-1-3000.pth")
    # cfg.MODEL.WEIGHTS = os.path.join('results/v9', "model_final.pth")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0069999.pth")  # this is the path of the trained vinbigdata model
    cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 200000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(category)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False # include empty annotation images




    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    print('start training')
    # trainer.train()
    print('finish training')

    # testing plot
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join('results/vindr-online', "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.15   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)


    dataset_dicts = get_dicts("test")
    vindr_metadata = MetadataCatalog.get("vindr_test")
    plot = True
    if plot:
        # output plotted images files
        # path = '/home/xinyue/original_image'
        # filename = os.listdir(path)
        # for file in filename:
        #     file_path = os.path.join(path,file)
        #     im = cv2.imread(file_path)
        #     outputs = predictor(
        #         im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        #     v = Visualizer(im[:, :, ::-1],
        #                    metadata=vindr_metadata,
        #                    scale=1.0,
        #                    )
        #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #     plt.figure(dpi=300)
        #     plt.imshow(out.get_image()[:, :, ::-1])
        #     str = 'Prediction'
        #     plt.title(str)
        #     plt.savefig('/home/xinyue/predicted_images/predicted_'+file[:-4]+'.png')
        #     # plt.show()

        for d in random.sample(dataset_dicts, 10):
            if d['annotations'] == []:
                continue

            go = False
            for i in range(len(d['annotations'])):
                if d['annotations'][i]['category_id'] == 17:
                    go = True
            if not go:
                continue

            print(d["file_name"])
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                           metadata=vindr_metadata,
                           scale=1.0,
            )

            # outputs = filter_classes(outputs, category, name = 'Lung')

            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.figure(dpi=300)
            plt.imshow(out.get_image()[:, :, ::-1])
            str = 'Prediction'
            plt.title(str)
            plt.show()


            v = Visualizer(im[:, :, ::-1],
                           metadata=vindr_metadata,
                           scale=1.0,
                           )

            out = v.draw_dataset_dict(d)
            plt.figure(dpi=300)
            plt.imshow(out.get_image())
            str = 'GroundTruth'
            plt.title(str)
            plt.show()

            # d = filter_classes(d, category, name = 'Other lesion')
            # v = Visualizer(im[:, :, ::-1],
            #                metadata=vindr_metadata,
            #                scale=1.0,
            #                )
            # out = v.draw_dataset_dict(d)
            # plt.figure(dpi=300)
            # plt.imshow(out.get_image())
            # str = 'GroundTruth'
            # plt.title(str)
            # plt.show()

    # evaluate
    evaluator = VinbigdataEvaluator("vindr_test", output_dir="./results/vindr")
    # evaluator = COCOEvaluator("vindr_test", output_dir="./results/vindr")
    val_loader = build_detection_test_loader(cfg, "vindr_test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == '__main__':
    # my_predictor()
    run()
    # this one is deprecated because doesn't have augmentation and transformation. use train-vindr-online.py