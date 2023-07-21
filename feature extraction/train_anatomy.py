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
from detectron2.data import build_detection_test_loader
from tqdm import tqdm
from evaluator import VinbigdataEvaluator
import pandas as pd

CUDA_LAUNCH_BLOCKING=1

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
            # if idx % 50 == 0:
            #     print('{} loaded'.format(idx))
            image_id_ori = image_id
    record["annotations"] = objs
    dataset_dicts.append(record)
    print('vinbigdata_dicts loaded')

    with open("data_dicts.pkl", "wb") as tf:
        pickle.dump(dataset_dicts, tf)
        print('dicts saved')
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

def get_mimic_box_coord(x1,y1,x2,y2,mimic_shape, image_id, original_width, original_height):
    top, bottom, left, right, ratio = get_Ratio(mimic_shape[image_id])
    coordinates = [x1, x2, y1, y2]
    scales = top, bottom, left, right, ratio
    original_x1, original_x2, original_y1, original_y2 = get_Original_Coordinates(coordinates, scales)

    x1 = original_x1 * (1024 / original_width)
    x2 = original_x2 * (1024 / original_width)
    y1 = original_y1 * (1024 / original_height)
    y2 = original_y2 * (1024 / original_height)
    return x1,y1,x2,y2

def get_mimic_ana_dicts(dataset_dir = '/home/xinyue/dataset/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/scene_graph'): # ordered csv
    if exists('dictionary/mimic_ana_dicts.pkl'):
        with open('dictionary/mimic_ana_dicts.pkl', "rb") as tf:
            data_dicts = pickle.load(tf)
        return data_dicts
    with open('dictionary/mimic_shape_full.pkl', 'rb') as f:
        mimic_shape = pickle.load(f)
    mimic_shape = convert_shape(mimic_shape)
    data_dicts = []
    json_files = sorted(os.listdir(dataset_dir))
    category_set = {}

    n = 0
    for idx in tqdm(range(len(json_files))):
        json_file = json_files[idx]
    # for idx, json_file in enumerate(json_files):
        record = {}
        objs = []
        path = os.path.join(dataset_dir, json_file)
        with open(path) as f:
            data = json.load(f)
            record['file_name'] = '/home/xinyue/dataset/mimic-cxr-png/' + data['image_id'] + '.png'
            record['image_id'] = idx
            record['height'] = 1024
            record['width'] = 1024
            for object in data['objects']:
                # try:
                #     ratio = object['width']/object['original_width']
                # except:
                #     n += 1

                original_x1 = object['original_x1']
                original_y1 = object['original_y1']
                original_x2 = object['original_x2']
                original_y2 = object['original_y2'] # this is not accurate, deprecated. use the code below
                try:
                    # (original_height, original_width) = mimic_shape[data['image_id']]
                    # original_width = object['original_width']
                    # original_height = object['original_height']
                    # if original_height == 0 or original_width == 0:
                    (original_height, original_width) = mimic_shape[data['image_id']]
                except:
                    n+= 1
                    continue
                x1 = object['x1']
                y1 = object['y1']
                x2 = object['x2']
                y2 = object['y2']
                top, bottom, left, right, ratio = get_Ratio(mimic_shape[data['image_id']])
                coordinates = [x1, x2, y1, y2]
                scales = top, bottom, left, right, ratio
                original_x1, original_x2, original_y1, original_y2 = get_Original_Coordinates(coordinates, scales)

                x1 = original_x1 * (1024/original_width)
                x2 = original_x2 * (1024/original_width)
                y1 = original_y1 * (1024/original_height)
                y2 = original_y2 * (1024/original_height)

                # x1 = original_x1
                # x2 = original_x2
                # y1 = original_y1
                # y2 = original_y2

                if object['name'] not in category_set:
                    category_set[object['name']] = len(category_set)
                obj = {
                    'bbox': [x1,y1,x2,y2],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    "category_id": category_set[object['name']]
                }
                objs.append(obj)
            record['annotations'] = objs
        data_dicts.append(record)
    print('total skipped', n)
    with open("dictionary/category_ana.pkl", "wb") as tf:
        pickle.dump(category_set, tf)
        print('dicts saved')
    with open("dictionary/mimic_ana_dicts.pkl", "wb") as tf:
        pickle.dump(data_dicts, tf)
        print('dicts saved')
    return data_dicts

def find_report(study_id, subject_id):
    report_path = '/home/xinyue/dataset/mimic_reports'
    p1 = 'p' + str(subject_id)[:2]
    p2 = 'p'+str(subject_id)
    report_name = 's' + str(int(study_id)) + '.txt'
    with open(os.path.join(report_path, p1, p2, report_name), 'r') as f:
        report = f.read()

    report.replace('\n', '').replace('FINDINGS', '\nFINDINGS').replace('IMPRESSION', '\nIMPRESSION')
    return report

target_sets = {}
def target_image(dicom_id, df_all):
    # select the image that has/doesn't have diffuse diseases
    want_diffuse = True # if want "has": True; doesn't have: False
    try:
        study_id = df_all[df_all['dicom_id'] == dicom_id]['study_id'].values[0]
        subject_id = df_all[df_all['dicom_id'] == dicom_id]['subject_id'].values[0]
        report = find_report(study_id, subject_id)
    except:
        return not want_diffuse
    if 'interstitial edema' in report.replace('\n', ' ').replace('  ', ' ').lower():
        target_sets[dicom_id] = want_diffuse
    else:
        target_sets[dicom_id] = not want_diffuse
    return target_sets[dicom_id]

def get_mimic_ana_gold_dicts(dataset_dir = '/home/xinyue/dataset/physionet.org/files/chest-imagenome/1.0.0/gold_dataset/gold_bbox_coordinate_annotations_1000images.csv', if_filter = False): # ordered csv
    # for the gold standard dataset
    with open('/home/xinyue/VQA_ReGat/data/mimic/mimic_shape_full.pkl', 'rb') as f:
        mimic_shape = pickle.load(f)
    mimic_shape = convert_shape(mimic_shape)

    if if_filter:
        all_path = '/home/xinyue/dataset/mimic/mimic_all.csv'
        df_all = pd.read_csv(all_path)

    kg = get_kg2()
    label2id = {}
    for item in kg:
        item = item.lower()
        label2id[item] = len(label2id)
    csv_file = dataset_dir
    with open(csv_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        reader = csv.DictReader(csvfile)
        dataset_dicts = []
        image_id_ori = '0'
        i = 0
        objs = []
        for idx, row in enumerate(tqdm(reader)):
            image_id = row['image_id'][:-4]
            if if_filter:
                if image_id in target_sets:
                    if target_sets[image_id] == False:
                        continue
                else:
                    if not target_image(image_id, df_all):
                        continue
            if image_id != image_id_ori:
                if len(objs) > 0:
                    record["annotations"] = objs
                    dataset_dicts.append(record)

                record = {}
                objs = []

                filename = 'train/' + image_id + '.dicom'
                # ds = pydicom.dcmread(os.path.join(dataset_dir, filename))
                filename = '/home/xinyue/dataset/mimic-cxr-png/' + image_id + '.png'
                (width, height) = mimic_shape[image_id]
                record["file_name"] = filename
                record["image_id"] = i
                record["height"] = 1024
                record["width"] = 1024

                if row['original_x1'] != '':
                    x1 = float(row['original_x1']) * (1024 / width)
                    y1 = float(row['original_y1']) * (1024 / height)
                    x2 = float(row['original_x2']) * (1024 / width)
                    y2 = float(row['original_y2']) * (1024 / height)


                    # x1 = float(row['x1'])
                    # y1 = float(row['y1'])
                    # x2 = float(row['x2'])
                    # y2 = float(row['y2'])
                    # x1, y1, x2, y2, = get_mimic_box_coord(x1, y1, x2, y2, mimic_shape, image_id,original_width, original_height)

                    if x1 > x2 or y1 > y2:
                        i += 1
                        continue

                    obj = {
                        "bbox": [x1, y1, x2, y2] if row['original_x1'] != '' else '',
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": int(label2id[row['bbox_name']]),
                    }
                    objs.append(obj)
                i += 1
            else:
                if row['original_x1'] != '':
                    x1 = float(row['original_x1']) * (1024 / width)
                    y1 = float(row['original_y1']) * (1024 / height)
                    x2 = float(row['original_x2']) * (1024 / width)
                    y2 = float(row['original_y2']) * (1024 / height)

                    # x1 = float(row['x1'])
                    # y1 = float(row['y1'])
                    # x2 = float(row['x2'])
                    # y2 = float(row['y2'])
                    # try:
                    #     x1, y1, x2, y2, = get_mimic_box_coord(x1, y1, x2, y2, mimic_shape, image_id,
                    #                                           float(row['original_width']), float(row['original_height']))
                    # except:
                    #     image_id_ori = image_id
                    #     continue
                    if x1 > x2 or y1 > y2:
                        image_id_ori = image_id
                        continue
                    obj = {
                        "bbox": [x1, y1, x2, y2],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": int(label2id[row['bbox_name']]),
                    }
                    objs.append(obj)

            # record["annotations"] = objs
            #
            # dataset_dicts.append(record)
            # if i % 50 == 0:
            #     print('{} loaded'.format(idx))
            image_id_ori = image_id
    record["annotations"] = objs
    dataset_dicts.append(record)
    print("mimic ana gold dicts loaded")

    # with open("data_dicts-vindr", "wb") as tf:
    #     pickle.dump(dataset_dicts, tf)
    #     print('dicts saved')
    return dataset_dicts


def get_dicts(split, gold_weights):
    # data_dicts = get_vinbigdata_dicts('/home/xinyue/dataset/vinbigdata-png')
    if gold_weights:
        data_dicts = get_mimic_ana_gold_dicts()
    else:
        data_dicts = get_mimic_ana_dicts('/home/xinyue/dataset/physionet.org/files/chest-imagenome/1.0.0/silver_dataset/scene_graph/')



    l_train_dicts = int(len(data_dicts) * 0.8)
    l_val_dicts = (int(len(data_dicts)) - l_train_dicts)
    # l_test_dicts = int(len(data_dicts)) - l_train_dicts - l_val_dicts
    train_dicts = data_dicts[:l_train_dicts]
    val_dicts = data_dicts[l_train_dicts: l_train_dicts + l_val_dicts]
    # test_dicts = data_dicts[-l_test_dicts:]
    if split == 'train':
        return data_dicts
        # return train_dicts
    elif split == 'val':
        return val_dicts
    elif split == 'full':
        return data_dicts
    # elif split == 'test':
    #     return test_dicts
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
    kg_dict['Apical zone of Left lung'] = 'Lung'
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

def get_kg2():
    # more simpler names
    kg_dict = {}
    # anatomical part
    kg_dict['right lung'] = 'Lung'
    kg_dict['right upper lung zone'] = 'Lung'
    kg_dict['right mid lung zone'] = 'Lung'
    kg_dict['right lower lung zone'] = 'Lung'
    kg_dict['right hilar structures'] = 'Lung'
    kg_dict['right apical zone'] = 'Lung'
    kg_dict['right costophrenic angle'] = 'Pleural'
    kg_dict['right hemidiaphragm'] = 'Pleural' # probably
    kg_dict['left lung'] = 'Lung'
    kg_dict['left upper lung zone'] = 'Lung'
    kg_dict['left mid lung zone'] = 'Lung'
    kg_dict['left lower lung zone'] = 'Lung'
    kg_dict['left hilar structures'] = 'Lung'
    kg_dict['left apical zone'] = 'Lung'
    kg_dict['left costophrenic angle'] = 'Pleural'
    kg_dict['left hemidiaphragm'] = 'Pleural' # probably

    kg_dict['trachea'] = 'Lung'
    kg_dict['right clavicle'] = 'Bone'
    kg_dict['left clavicle'] = 'Bone'
    kg_dict['aortic arch'] = 'Heart'
    kg_dict['upper mediastinum'] = 'Mediastinum'
    kg_dict['svc'] = 'Heart'
    kg_dict['cardiac silhouette'] = 'Heart'
    kg_dict['cavoatrial junction'] = 'Heart'
    kg_dict['right atrium'] = 'Heart'
    kg_dict['carina'] = 'Lung'

    return kg_dict

def filter_classes(outputs, category, gold_weights, name = None):
    if name == None:
        return outputs
    target_labels = set()

    if gold_weights:
        kg_dict = get_kg2()
    else:
        kg_dict = get_kg()
    target_names = []
    for item in kg_dict:
        if kg_dict[item] == name:
            target_names.append(item)
    # print(target_names)

    target_names = name


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


def run(gold_weights):
    kg = get_kg2()
    for key, val in enumerate(kg):
        print(val)

    if gold_weights:
        d = get_kg2()
        category = {}
        for item in d:
            category[item] = len(category)
    else:
        with open('dictionary/category_ana.pkl', "rb") as tf:
            category = pickle.load(tf)


    for d in ["train", "val"]:
        DatasetCatalog.register("mimic_ana_" + d, lambda d=d: get_dicts(d,gold_weights))
        MetadataCatalog.get("mimic_ana_" + d).set(thing_classes=list(category))
    mimic_ana_metadata = MetadataCatalog.get("mimic_ana_train")


    dataset_dicts = get_dicts('train',gold_weights)
    for d in random.sample(dataset_dicts, 1):
    # for d in dataset_dicts[:20]:
        print(d["file_name"])

        if d['annotations'] == []:
            continue

        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=mimic_ana_metadata, scale=1)
        out = visualizer.draw_dataset_dict(d)

        plt.figure(dpi=300)
        plt.imshow(out.get_image())
        plt.show()


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    if gold_weights:
        cfg.OUTPUT_DIR = 'results/anatomy_gold'
        cfg.MODEL.WEIGHTS = os.path.join('results/anatomy_gold', "model_final.pth") # this is the path of the trained vinbigdata model
        # cfg.MODEL.WEIGHTS = os.path.join('results/anatomy_gold', "model_0039999.pth.pth") # this is the path of the trained vinbigdata model
    else:
        cfg.OUTPUT_DIR = 'results/anatomy'
        cfg.MODEL.WEIGHTS = os.path.join('results/anatomy', "model_final.pth")  # this is the path of the trained vinbigdata model
    cfg.DATASETS.TRAIN = ("mimic_ana_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-1-3000.pth")
    cfg.SOLVER.IMS_PER_BATCH = 2
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 40000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
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
    # cfg.MODEL.WEIGHTS = os.path.join('results/v9', "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)


    dataset_dicts = get_dicts("val",gold_weights)
    plot = False
    # filter_list = ['Right lung', 'Hilar Area of the Right Lung', 'right costophrenic sulcus;;Right costodiaphragmatic recess', 'Right hemidiaphragm', 'Left lung', 'Hilar Area of the Left Lung', 'left costophrenic sulcus;;Left costodiaphragmatic recess', 'Left hemidiaphragm', 'Trachea&&Main Bronchus', 'Vertebral column', 'Aortic arch structure', 'Mediastinum', 'Cardiac shadow viewed radiologically;;Heart', 'cavoatrial', 'Descending aorta' ]
    # filter_list = [  'right costophrenic sulcus;;Right costodiaphragmatic recess', 'Right hemidiaphragm', 'left costophrenic sulcus;;Left costodiaphragmatic recess', 'Left hemidiaphragm', 'Trachea&&Main Bronchus',  'Aortic arch structure',  'cavoatrial', 'Descending aorta' ]
    filter_list = ['left costophrenic angle','right costophrenic angle', 'cavoatrial junction', 'carina','left hemidiaphragm', 'right hemidiaphragm']
    filter_list = ['Mediastinum', 'Superior mediastinum']

    if plot:
        for d in random.sample(dataset_dicts, 10):
        # for d in dataset_dicts[:10]:
            if d['annotations'] == []:
                continue
            # temporary testing
            d["file_name"] = '/home/xinyue/dataset/mimic-cxr-png/df325ab0-419a2647-8a4a42d0-5dbe555e-c451d261.png'
            # d["file_name"] = '/home/xinyue/dataset/mimic-cxr-png/02255881-809e6282-9f5742e5-5f7da63c-9d183812.png'
            print(d["file_name"])
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                           metadata=mimic_ana_metadata,
                           scale=1.0,
            )

            # outputs = filter_classes(outputs, category, gold_weights, name = filter_list)

            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            plt.figure(dpi=300)
            plt.imshow(out.get_image()[:, :, ::-1])
            str = 'Prediction'
            plt.title(str)
            plt.show()


            v = Visualizer(im[:, :, ::-1],
                           metadata=mimic_ana_metadata,
                           scale=1.0,
                           )

            # d = filter_classes(d, category, gold_weights, name = filter_list)

            out = v.draw_dataset_dict(d)
            plt.figure(dpi=300)
            plt.imshow(out.get_image())
            str = 'GroundTruth'
            plt.title(str)
            plt.show()

    # evaluate
    evaluator = VinbigdataEvaluator("mimic_ana_val", output_dir=cfg.OUTPUT_DIR)
    # evaluator = COCOEvaluator("mimic_ana_val", output_dir="./results/anatomy")
    val_loader = build_detection_test_loader(cfg, "mimic_ana_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == '__main__':
    # my_predictor()

    clear_cach = True
    if clear_cach:
        # remove results/anatomy_gold/mimic_ana_val_coco_format.json
        if os.path.exists('results/anatomy_gold/mimic_ana_val_coco_format.json'):
            os.remove('results/anatomy_gold/mimic_ana_val_coco_format.json')

    gold_weights = True
    run(gold_weights)