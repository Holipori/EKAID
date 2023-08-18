import h5py
import pickle
import numpy as np
from tqdm import tqdm
from ana_bbox_generator import get_adj_matrix
import pandas as pd
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
def get_kg_ana_only():
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
    kg_dict['Edema'] = 'Lung'

    return kg_dict

def get_kg():
    # anatomical part
    kg_dict = get_kg_ana_only()

    # disease part
    kg_dict['Aortic enlargement'] = 'Heart'
    kg_dict['Atelectasis'] = 'Lung'
    kg_dict['Calcification'] = 'Bone'
    kg_dict['Cardiomegaly'] = 'Heart'
    kg_dict['Consolidation'] = 'Lung'
    kg_dict['ILD'] = 'Lung'
    kg_dict['Infiltration'] = 'Lung'
    kg_dict['Lung Opacity'] = 'Lung'
    kg_dict['Nodule/Mass'] = 'Lung'
    kg_dict['Other lesion'] = 'Lung'
    kg_dict['Pleural effusion'] = 'Pleural'
    kg_dict['Pleural thickening'] = 'Pleural'
    kg_dict['Pneumothorax'] = 'Pleural'
    kg_dict['Pulmonary fibrosis'] = 'Lung'
    kg_dict['Clavicle fracture'] = 'Bone'
    kg_dict['Emphysema'] = 'Lung'
    kg_dict['Enlarged PA'] = 'Heart'
    kg_dict['Lung cavity'] = 'Lung'
    kg_dict['Lung cyst'] = 'Lung'
    kg_dict['Mediastinal shift'] = 'Mediastinum'
    kg_dict['Rib fracture'] = 'Bone'
    kg_dict['Fracture'] = 'Bone'

    return kg_dict

def cmb_pred_classes(pred_classes_ana, pred_classes_loc, ana_category):
    # pred_classes_di += len(ana_category)
    pred_classes_loc += len(ana_category)

    # thing_classes = ana_thing_classes + di_thing_classes
    pred_classes = np.hstack((pred_classes_ana, pred_classes_loc))

    return pred_classes
def get_semantic_adj(pred_classes_ana, pred_classes_loc, ana_thing_classes, di_thing_classes,kg_ana, small_adj, small_name2index):
    '''

    :param pred_classes_ana: 36
    :param pred_classes_di: 25
    :return:
    '''

    # pred_classes_di += len(ana_thing_classes)
    pred_classes_loc += len(ana_thing_classes)

    # for i in range(len(di_thing_classes)):
    #     # di_thing_classes[i] = di_thing_classes[i].lower().replace(' ', '_')
    #     if 'fracture' in di_thing_classes[i]:
    #         di_thing_classes[i] = 'fracture'

    thing_classes = ana_thing_classes + di_thing_classes
    pred_classes = np.hstack((pred_classes_ana,pred_classes_loc))

    ana_thing_classes_set = set(ana_thing_classes)
    di_thing_classes_set = set(di_thing_classes)
    adj_matrix = np.zeros([100, 100], int)
    for i in range(test_topk_per_image):
        for j in range(i,test_topk_per_image):
            if pred_classes[i] == len(thing_classes) or pred_classes[j] == len(thing_classes):
                continue
            if kg_ana[thing_classes[pred_classes[i]]] == kg_ana[thing_classes[pred_classes[j]]]:
                if thing_classes[pred_classes[i]] in ana_thing_classes_set and thing_classes[
                    pred_classes[j]] in di_thing_classes_set or thing_classes[
                    pred_classes[j]] in ana_thing_classes_set and thing_classes[
                    pred_classes[i]] in di_thing_classes_set:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

            # semantic garph 2
            if thing_classes[pred_classes[i]].lower() in small_name2index and thing_classes[
                pred_classes[j]].lower() in small_name2index:
                # small_adj is the 14x14 co-occurrence matrix
                value = max(small_adj[small_name2index[thing_classes[pred_classes[i]].lower()], small_name2index[
                    thing_classes[pred_classes[j]].lower()]], adj_matrix[i, j])
                adj_matrix[i, j] = value
                adj_matrix[j, i] = value
    # adj_matrix[test_topk_per_image,:test_topk_per_image]  = 1
    # adj_matrix[:test_topk_per_image, test_topk_per_image] = 1

    return adj_matrix

def get_countingAdj_name2index():
    path = '/home/qiyuan/2021summer/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv.gz'
    df = pd.read_csv(path)

    mimic_list = df.columns[2:16].values

    ans2label = {key.lower(): i for i, key in enumerate(mimic_list)}
    return ans2label

def save_h5(final_features, bboxes, adj_matrix, test_topk_per_image, pred_classes, semantic_adj, times=0, length = 100):

    filename = './output/cmb_bbox_di_feats.hdf5'

    if times == 0:
        h5f = h5py.File(filename, 'w')
        image_features_dataset = h5f.create_dataset("image_features", (length, test_topk_per_image, 1024),
                                                    maxshape=(None, test_topk_per_image, 1024),
                                                    chunks=(100, test_topk_per_image, 1024),
                                                    dtype='float32')
        image_bb_dataset = h5f.create_dataset("image_bb", (length, test_topk_per_image, 4),
                                              maxshape=(None, test_topk_per_image, 4),
                                              chunks=(100, test_topk_per_image, 4),
                                              dtype='float32')
        image_adj_matrix_dataset = h5f.create_dataset("image_adj_matrix", (length, 100, 100),
                                                      maxshape=(None, 100, 100),
                                                      chunks=(100, 100, 100),
                                                      dtype='int64')
        semantic_adj_matrix_dataset = h5f.create_dataset("semantic_adj_matrix", (length, 100, 100),
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
        image_bb_dataset = h5f['image_bb']
        image_adj_matrix_dataset = h5f['image_adj_matrix']
        semantic_adj_matrix_dataset = h5f['semantic_adj_matrix']
        bbox_label_dataset = h5f['bbox_label']

    if len(final_features) != length:
        adding = len(final_features)
    else:
        adding = length

    image_features_dataset.resize([times*length+adding, test_topk_per_image, 1024])
    image_features_dataset[times*length:times*length+adding] = final_features

    image_bb_dataset.resize([times*length+adding, test_topk_per_image, 4])
    image_bb_dataset[times*length:times*length+adding] = bboxes

    image_adj_matrix_dataset.resize([times*length+adding, 100, 100])
    image_adj_matrix_dataset[times*length:times*length+adding] = adj_matrix

    semantic_adj_matrix_dataset.resize([times * length + adding, 100, 100])
    semantic_adj_matrix_dataset[times * length:times * length + adding] = semantic_adj

    bbox_label_dataset.resize([times * length + adding, test_topk_per_image])
    bbox_label_dataset[times * length:times * length + adding] = pred_classes

    h5f.close()

path1 = './output/mimic_ana_box/ana_bbox_features_full.hdf5'
# path2 = '/home/xinyue/faster-rcnn/output/mimic_box_coords/bbox_disease_features_by_coords.hdf5'
path3 = './output/mimic_disease_box_by_location/bbox_features_disease_by_location.hdf5'
hf1 = h5py.File(path1, 'r')
# hf2 = h5py.File(path2, 'r')
hf3 = h5py.File(path3, 'r')

test_topk_per_image = 2*hf1['image_features'].shape[1]


di_thing_classes = list(get_vindr_label2id())
di_thing_classes = [di_thing_classes[i].lower() for i in range(len(di_thing_classes))]
ana_thing_classes = list(get_kg_ana_only())
ana_thing_classes = [ana_thing_classes[i].lower() for i in range(len(ana_thing_classes))]

# semantic graph 2: co-occurance
with open('dictionary/GT_counting_adj.pkl', "rb") as tf:
    small_counting_adj = pickle.load(tf)
    for i in range(len(small_counting_adj)):
        small_counting_adj[i]  = small_counting_adj[i]/small_counting_adj[i][i]
    small_counting_adj = np.where(small_counting_adj > 0.18, 2, 0) # co-occurrence kg
small_name2index = get_countingAdj_name2index()
# for more information, please refer to faster-rcnn folder combine_dicts.py



# semantic graph 1: anatomical
kg_ana = get_kg() # ana_KG
new_kg = {}
for key in kg_ana:
    new_kg[key.lower()] = kg_ana[key]
kg_ana = new_kg


final_features = []
bboxes = []
adj_matrix = []
pred_classes = []
semantic_adj = []
length = 5000
times = 0
resume = False # remember to check before running
if resume:
    stopped_batch_num = 65000  # the number you see in the terminal when stooped
    stopped_img_num = stopped_batch_num
    continue_i = stopped_img_num - length
    times = int((stopped_img_num - length)/length)
for i in tqdm(range(len(hf1['image_features']))):
    if resume:
        if i < continue_i:
            continue
    final_features.append(np.vstack((hf1['image_features'][i], hf3['image_features'][i])))
    bboxes.append(np.vstack((hf1['image_bb'][i],  hf3['image_bb'][i])))
    pred_classes.append(cmb_pred_classes(hf1['bbox_label'][i],  hf3['bbox_label'][i], ana_thing_classes))
    semantic_adj.append(get_semantic_adj(hf1['bbox_label'][i], hf3['bbox_label'][i], ana_thing_classes, di_thing_classes, kg_ana, small_counting_adj, small_name2index))

    if len(final_features) == length or i == len(hf1['image_adj_matrix']) - 1:
        final_features = np.array(final_features)
        bboxes = np.array(bboxes)
        pred_classes = np.array(pred_classes)
        semantic_adj = np.array(semantic_adj)
        adj_matrix = get_adj_matrix(bboxes)
        save_h5(final_features, bboxes, adj_matrix, test_topk_per_image, pred_classes,semantic_adj, times=times, length=length)
        final_features = []
        bboxes = []
        adj_matrix = []
        pred_classes = []
        semantic_adj = []
        times += 1

hf1.close()
# hf2.close()
hf3.close()