import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import torch

def find_dicom(id, d2):
    # find dicom_id from d2 that has the same study_id and ViewCodeSequence_CodeMeaning is in ['postero-anterior', 'antero-posterior']
    dicoms = d2[d2['study_id'] == id]
    dicom = dicoms[dicoms['ViewCodeSequence_CodeMeaning'].isin(['postero-anterior', 'antero-posterior'])]
    if len(dicom) == 0:
        dicom = dicoms
    dicom_id = dicom['dicom_id'].values[0]
    view = dicom['ViewCodeSequence_CodeMeaning'].values[0]
    # dicom_id = dicom_id.values[0]['dicom_id']
    return dicom_id, view
    # while 1:
    #     if d2.iloc[j]['study_id'] != id and d2.iloc[j]['']:
    #         j += 1
    #     else:
    #         return d2.iloc[j]['dicom_id'], d2.iloc[j]['ViewCodeSequence_CodeMeaning'], j
def find_split(dicom_id, d3):
    # find 'split' from d3 that has the same dicom_id
    split = d3[d3['dicom_id'] == dicom_id]['split'].values[0]
    return split
    # while 1:
    #     if d3.iloc[j]['study_id'] != id:
    #         j += 1
    #     else:
    #         return d3.iloc[j]['split']

def purify(array):
    for i in range(len(array)):
        if np.isnan(array[i]) or array[i] == -1:
            array[i] = 0
    return array

def add_finding(lib, finding, study_id):
    finding = tuple(finding)
    if finding not in lib:
        lib[finding] = [study_id]
    else:
        lib[finding].append(study_id)
    return lib

def get_caption(adding, dropping,):
    if len(adding) == 1:
        output1 = 'image B has an additional finding of'
    elif len(adding) > 1:
        output1 = 'image B has additional findings of'
    elif len(adding) == 0:
        output1 = ''
    for item in adding:
        if item == adding[-1] and len(adding) != 1:
            output1 = output1 + ' and ' + item
        else:
            if len(adding) == 1:
                output1 = output1 + ' ' + item
            else:
                output1 = output1 + ' ' + item + ','
    if len(adding) != 0:
        output1 = output1 + ' than image A. '

    if len(dropping) == 1:
        output2 = 'image B is missing the finding of'
    elif len(dropping) > 1:
        output2 = 'image B is missing the findings of'
    elif len(dropping) == 0:
        output2 = ''
    for item in dropping:
        if item == dropping[-1] and len(dropping) != 1:
            output2 = output2 + ' and ' + item
        else:
            if len(dropping) == 1:
                output2 = output2 + ' ' + item
            else:
                output2 = output2 + ' ' + item + ','

    if len(dropping) != 0:
        output2 = output2 + ' than image A. '
    return output1 + output2

def add_stu_sub(sub_stu, subject_id, study_id):
    if study_id not in sub_stu:
        sub_stu[study_id] = [subject_id]
    else:
        sub_stu[study_id].append(subject_id)
    return sub_stu

def get_label(caption_list, max_seq):
    output = np.zeros(max_seq)
    output[:len(caption_list)] = np.array(caption_list)
    return output

def get_study2dicom():
    path = '/home/xinyue/dataset/mimic/mimic_all.csv'
    df = pd.read_csv(path)
    dict = {}
    for i in tqdm(range(len(df))):
        study = df.iloc[i]['study_id']
        dicom = df.iloc[i]['dicom_id']
        dict[study] = dicom
    path2 = '/home/xinyue/dataset/mimic/study2dicom.pkl'
    with open(path2, "wb") as tf:
        pickle.dump(dict, tf)
        print('dicts saved')

def dicom2id():
    # the id that in features hf.
    with open('/home/xinyue/VQA_ReGat/data/mimic/mimic_shape_full.pkl', 'rb') as f:
        mimic_shape = pickle.load(f)
    dict = {}
    for i in tqdm(range(len(mimic_shape))):
        dict[mimic_shape[i]['image']] = i
    path2 = '/home/xinyue/dataset/mimic/dicom2id.pkl'
    with open(path2, "wb") as tf:
        pickle.dump(dict, tf)
        print('dicts saved')
def torch_broadcast_adj_matrix(adj_matrix, label_num=11, device=torch.device("cuda")):
    """ broudcast spatial relation graph

    Args:
        adj_matrix: [batch_size,num_boxes, num_boxes]

    Returns:
        result: [batch_size,num_boxes, num_boxes, label_num]
    """
    result = []
    for i in range(1, label_num+1):
        index = torch.nonzero((adj_matrix == i).view(-1).data).squeeze()
        curr_result = torch.zeros(
            adj_matrix.shape[0], adj_matrix.shape[1], adj_matrix.shape[2]).to(device)
        curr_result = curr_result.view(-1)
        curr_result[index] += 1
        result.append(curr_result.view(
            (adj_matrix.shape[0], adj_matrix.shape[1],
             adj_matrix.shape[2], 1)))
    result = torch.cat(result, dim=3)
    return result

def process_matrix(adj_matrix, cfg, num_objects, device, type):
    adj_matrix = adj_matrix.to(device)
    adj_matrix = adj_matrix[:, :num_objects, :num_objects]
    if type == 'spatial':
        label_num = cfg.model.change_detector.spa_label_num
    elif type == 'semantic':
        label_num = cfg.model.change_detector.sem_label_num
    adj_matrix = torch_broadcast_adj_matrix(adj_matrix, label_num=label_num, device=device).to(device)
    return adj_matrix


def torch_extract_position_matrix(bbox, nongt_dim=36):
    """ Extract position matrix

    Args:
        bbox: [batch_size, num_boxes, 4]

    Returns:
        position_matrix: [batch_size, num_boxes, nongt_dim, 4]
    """

    xmin, ymin, xmax, ymax = torch.split(bbox, 1, dim=-1)
    # [batch_size,num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [batch_size,num_boxes, num_boxes]
    delta_x = center_x-torch.transpose(center_x, 1, 2)
    delta_x = torch.div(delta_x, bbox_width)

    delta_x = torch.abs(delta_x)
    threshold = 1e-3
    delta_x[delta_x < threshold] = threshold
    delta_x = torch.log(delta_x)
    delta_y = center_y-torch.transpose(center_y, 1, 2)
    delta_y = torch.div(delta_y, bbox_height)
    delta_y = torch.abs(delta_y)
    delta_y[delta_y < threshold] = threshold
    delta_y = torch.log(delta_y)
    delta_width = torch.div(bbox_width, torch.transpose(bbox_width, 1, 2))
    delta_width = torch.log(delta_width)
    delta_height = torch.div(bbox_height, torch.transpose(bbox_height, 1, 2))
    delta_height = torch.log(delta_height)
    concat_list = [delta_x, delta_y, delta_width, delta_height]
    for idx, sym in enumerate(concat_list):
        sym = sym[:, :nongt_dim]
        concat_list[idx] = torch.unsqueeze(sym, dim=3)
    position_matrix = torch.cat(concat_list, 3)
    return position_matrix

def torch_extract_position_embedding(position_mat, feat_dim, wave_length=1000,
                                     device=torch.device("cuda")):
    # position_mat, [batch_size,num_rois, nongt_dim, 4]
    feat_range = torch.arange(0, feat_dim / 8)
    dim_mat = torch.pow(torch.ones((1,))*wave_length,
                        (8. / feat_dim) * feat_range)
    dim_mat = dim_mat.view(1, 1, 1, -1).to(device)
    position_mat = torch.unsqueeze(100.0 * position_mat, dim=4)
    div_mat = torch.div(position_mat.to(device), dim_mat)
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    # embedding, [batch_size,num_rois, nongt_dim, 4, feat_dim/4]
    embedding = torch.cat([sin_mat, cos_mat], -1)
    # embedding, [batch_size,num_rois, nongt_dim, feat_dim]
    embedding = embedding.view(embedding.shape[0], embedding.shape[1],
                               embedding.shape[2], feat_dim)
    return embedding

if __name__ == '__main__':
    get_study2dicom()