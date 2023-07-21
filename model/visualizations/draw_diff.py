import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from utils.mimic_utils import purify
import random
import h5py
import sys
import cv2
import json
import torch
from utils.utils import load_checkpoint, decode_sequence, set_mode
from detectron2.utils.visualizer import Visualizer
from models.modules import ChangeDetector
# sys.path.append("..")
sys.path.append("../faster_rcnn")
# from train_vindr import my_predictor
from draw_single import find_report
from datasets.datasets import create_dataset
from configs.config import cfg, merge_cfg_from_file
import argparse
from utils.mimic_utils import process_matrix
import matplotlib.patches as patches
from models.dynamic_speaker_change_pos import DynamicSpeaker
from tqdm import tqdm



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
def move_adj(adj, len, mode='3to2'):
    # move the 3rd adj to 2nd position
    if mode == '3to2':
        adj[len:2 * len] = adj[2 * len:3 * len]
        adj[:, len:2 * len] = adj[:, 2 * len:3 * len]
    elif mode == '3to1':
        adj[:len] = adj[2 * len:3 * len]
        adj[:, :len] = adj[:, 2 * len:3 * len]
    return adj

def process_input(d_feature, q_feature, d_adj_matrix, q_adj_matrix,d_sem_adj_matrix, q_sem_adj_matrix,d_bb,q_bb, question):
    d_feature = torch.from_numpy(d_feature).unsqueeze(0).cuda()
    q_feature = torch.from_numpy(q_feature).unsqueeze(0).cuda()
    d_adj_matrix = d_adj_matrix.unsqueeze(0).cuda()
    q_adj_matrix = q_adj_matrix.unsqueeze(0).cuda()
    d_sem_adj_matrix = d_sem_adj_matrix.unsqueeze(0).cuda()
    q_sem_adj_matrix = q_sem_adj_matrix.unsqueeze(0).cuda()
    d_bb = torch.from_numpy(d_bb).unsqueeze(0).cuda()
    q_bb = torch.from_numpy(q_bb).unsqueeze(0).cuda()
    question = torch.from_numpy(question).unsqueeze(0).cuda()

    d_feature = d_feature.expand(64,d_feature.shape[1],d_feature.shape[2])
    q_feature = q_feature.expand(64,q_feature.shape[1],q_feature.shape[2])
    d_adj_matrix = d_adj_matrix.expand(64,d_adj_matrix.shape[1],d_adj_matrix.shape[2])
    q_adj_matrix = q_adj_matrix.expand(64,q_adj_matrix.shape[1],q_adj_matrix.shape[2])
    d_sem_adj_matrix = d_sem_adj_matrix.expand(64,d_sem_adj_matrix.shape[1],d_sem_adj_matrix.shape[2])
    q_sem_adj_matrix = q_sem_adj_matrix.expand(64,q_sem_adj_matrix.shape[1],q_sem_adj_matrix.shape[2])
    d_bb = d_bb.expand(64,d_bb.shape[1],d_bb.shape[2])
    q_bb = q_bb.expand(64,q_bb.shape[1],q_bb.shape[2])
    question = question.expand(64,question.shape[1])

    return d_feature, q_feature, d_adj_matrix, q_adj_matrix,d_sem_adj_matrix, q_sem_adj_matrix,d_bb,q_bb, question

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

def shift_bbx(bbx_labels, shift = 50):
    for i in range(len(bbx_labels)):
        for j in range(i+1,len(bbx_labels)):
            if abs(bbx_labels[i][1] - bbx_labels[j][1]) < shift:
                if bbx_labels[i][1] < bbx_labels[j][1]:
                    bbx_labels[i][1] -= shift
                    bbx_labels[j][1] += shift
                else:
                    bbx_labels[i][1] += shift
                    bbx_labels[j][1] -= shift
    return bbx_labels

def plotting_diff(input_idx = None, checkpoint_num = 34000):
    '''
    input_idx: the index of the data in dataset. If None, randomly go through the dataset
    '''

    # prepare dataset
    dataset_csv = './data/datasets/mimic_pair_questions.csv'
    df = pd.read_csv(dataset_csv)



    dataset_path = './data/datasets/VQA_mimic_dataset.h5'
    hf = h5py.File(dataset_path, 'r')
    feature_idx = hf['feature_idx']
    questions = hf['questions']
    answers = hf['answers']

    feature_path = './data/cmb_bbox_di_feats.hdf5'
    hf_feature = h5py.File(feature_path, 'r')
    features = hf_feature['image_features']
    bb = hf_feature['image_bb']
    bb_label = hf_feature['bbox_label']
    adj = hf_feature['image_adj_matrix']
    sem_adj = hf_feature['semantic_adj_matrix']

    results_file = './experiments/temp/%s/eval_sents/eval_results_%d.json' % (args.resume_fold, checkpoint_num)
    if not os.path.exists(results_file):
        results_file = results_file.replace('temp', 'final')
    with open(results_file, 'r') as f:
        results = json.load(f)
        # transform list of dictionary into a dict by image_id
        results_dict = {}
        for item in results:
            results_dict[item['image_id']] = item['caption']

    with open('/home/xinyue/dataset/mimic/study2dicom.pkl', 'rb') as f:
        study2dicom = pickle.load(f)

    dictionary_path = './data/datasets/vocab_mimic_VQA.json'
    with open(dictionary_path, 'r') as f:
        word_to_idx = json.load(f)
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    idx_to_word[0] = ' '
    print(word_to_idx)

    d = get_kg2()
    id2cat = {}
    for item in d:
        id2cat[len(id2cat)] = item
    id2dis = list(get_vindr_label2id())

    img_dir = '/home/xinyue/dataset/mimic-cxr-png'

    ##### input####
    idx_list = list(range(len(results_dict.keys())))
    random.shuffle(idx_list)

    if input_idx is not None:
        idx_list = [input_idx]
    for idx in tqdm(idx_list):
        if input_idx is not None:
            idx = input_idx
        else:
            idx = int(results[idx]['image_id'])
        # load data
        raw_question = df.iloc[idx]['question']
        raw_answer = df.iloc[idx]['answer'].strip()
        study_id = df.iloc[idx]['study_id']
        ref_id = df.iloc[idx]['ref_id']
        question_type = df.iloc[idx]['question_type']

        prediction = results_dict[str(idx)].replace(' ,', ',').replace(' .','.').strip()
        # prediction = 'no'

        ###### filtering ######
        if input_idx == None:
            if question_type != 'location':
                continue
            # if 'left' not in raw_question and 'right' not in raw_question:
            #     continue
            # if 'what abnormalities are' not in raw_question:
            #     continue
            # if ',' not in raw_answer:
            #     continue
            #
            # if 'what has changed in the left lung area' not in raw_question:
            #     continue
            #
            # if 'atelectasis' != raw_answer:
            #     continue
            # if 'cardiomegaly' not in raw_answer or 'edema' not in raw_answer:
            #     continue
            # if ',' not in raw_answer:
            #     continue
            # if 'yes' in raw_answer or 'no' in raw_answer:
            #     continue
            # if 'addition' not in raw_answer:
            #     continue
            if 'nothing' in raw_answer:
                continue
            if prediction != raw_answer:
                continue
            #
            # if 'nothing has changed' in prediction:
            #     continue
        ###### end filtering ######

        df_question = hf['questions'][idx]
        tranformed_question = [idx_to_word[w] for w in df_question]
        tranformed_question = ' '.join(tranformed_question).strip()

        # report
        report = find_report(study_id)
        print('main report: ', report)

        # loading feature
        node_one_num = int(len(features[0])/3)
        assert (node_one_num == 26)

        d1 = features[feature_idx[idx, 0]][:node_one_num]
        d2 = features[feature_idx[idx, 0]][-node_one_num:]
        d_feature = np.concatenate((d1, d2))
        d_bb1 = torch.from_numpy(bb[feature_idx[idx, 0]]).double()[:node_one_num]
        d_bb2 = torch.from_numpy(bb[feature_idx[idx, 0]]).double()[-node_one_num:]
        d_bb = np.concatenate((d_bb1, d_bb2))
        d_bb_label = bb_label[feature_idx[idx, 0]]

        q1 = features[feature_idx[idx, 1]][:node_one_num]
        q2 = features[feature_idx[idx, 1]][-node_one_num:]
        q_feature = np.concatenate((q1, q2))
        q_bb1 = torch.from_numpy(bb[feature_idx[idx, 1]]).double()[:node_one_num]
        q_bb2 = torch.from_numpy(bb[feature_idx[idx, 1]]).double()[-node_one_num:]
        q_bb = np.concatenate((q_bb1, q_bb2))
        q_bb_label = bb_label[feature_idx[idx, 1]]

        d_adj_matrix = torch.from_numpy(adj[feature_idx[idx, 0]]).double()
        q_adj_matrix = torch.from_numpy(adj[feature_idx[idx, 1]]).double()
        d_adj_matrix = move_adj(d_adj_matrix, node_one_num, mode='3to2')
        q_adj_matrix = move_adj(q_adj_matrix, node_one_num, mode='3to2')
        d_sem_adj_matrix = torch.from_numpy(sem_adj[feature_idx[idx, 0]]).double()
        q_sem_adj_matrix = torch.from_numpy(sem_adj[feature_idx[idx, 1]]).double()
        d_sem_adj_matrix = move_adj(d_sem_adj_matrix, node_one_num, mode='3to2')
        q_sem_adj_matrix = move_adj(q_sem_adj_matrix, node_one_num, mode='3to2')

        question = questions[idx]

        labels = answers[idx]
        labels = torch.from_numpy(labels).unsqueeze(0).cuda()




        # load checkpoint
        snapshot_dir = os.path.join('./experiments', 'temp', args.resume_fold, 'snapshots')
        snapshot_file = 'checkpoint_%d.pt' % checkpoint_num
        snapshot_full_path = os.path.join(snapshot_dir, snapshot_file)
        if not os.path.exists(snapshot_full_path):
            snapshot_full_path = snapshot_full_path.replace('temp', 'final')
        checkpoint = load_checkpoint(snapshot_full_path)
        change_detector_state = checkpoint['change_detector_state']
        speaker_state = checkpoint['speaker_state']
        # load model
        train_dataset, train_loader = create_dataset(cfg, 'train')
        change_detector = ChangeDetector(cfg, train_dataset.word_to_idx)
        change_detector.load_state_dict(change_detector_state)
        change_detector.to('cuda')

        speaker = DynamicSpeaker(cfg, len(train_dataset.get_idx_to_word()) + 1)
        speaker.load_state_dict(speaker_state)
        speaker.to('cuda')

        set_mode('eval', [change_detector, speaker])


        d_feature, q_feature, d_adj_matrix, q_adj_matrix,d_sem_adj_matrix, q_sem_adj_matrix,d_bb,q_bb, question = process_input(d_feature, q_feature, d_adj_matrix, q_adj_matrix,d_sem_adj_matrix, q_sem_adj_matrix,d_bb,q_bb, question)
        d_adj_matrix = process_matrix(d_adj_matrix,cfg, d_feature.shape[1], d_feature.device, type = 'spatial')
        q_adj_matrix = process_matrix(q_adj_matrix,cfg, q_feature.shape[1], q_feature.device, type = 'spatial')

        d_sem_adj_matrix = process_matrix(d_sem_adj_matrix, cfg, d_feature.shape[1], d_feature.device, type = 'semantic')
        q_sem_adj_matrix = process_matrix(q_sem_adj_matrix, cfg, q_feature.shape[1], q_feature.device, type = 'semantic')

        with torch.no_grad():
            chg_pos_logits, att_bef, att_aft, \
                    chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff = change_detector(d_feature, q_feature, d_adj_matrix, q_adj_matrix,d_sem_adj_matrix, q_sem_adj_matrix,d_bb,q_bb, question, setting = cfg.train.setting, graph=args.graph)



            speaker_output_pos, _ = speaker._sample(chg_pos_feat_bef,
                                                    chg_pos_feat_aft,
                                                    chg_pos_feat_diff,
                                                    labels, cfg)

            pos_dynamic_atts = speaker.get_module_weights().detach().cpu().numpy()  # (batch, seq_len, 3)

            gen_sents_pos = decode_sequence(idx_to_word, speaker_output_pos)
            # prediction = gen_sents_pos[0]
        d_bb = d_bb[0].squeeze()
        q_bb = q_bb[0].squeeze()


        # loading image
        name1 = study2dicom[int(study_id)] + '.png'
        path1 = os.path.join(img_dir,name1)
        img1 = cv2.imread(path1)
        name2 = study2dicom[int(ref_id)] + '.png'
        path2 = os.path.join(img_dir,name2)
        img2 = cv2.imread(path2)

        # add text
        output_text = 'Question: ' + raw_question
        output_text += '\n\n'
        output_text += 'GT Answer: ' + raw_answer
        output_text += '\n\n'
        output_text += 'Prediction: ' + prediction
        print(output_text)

        if len(att_bef.shape) == 3:
            iter_graph_num = 1
        elif len(att_bef.shape) == 4:
            iter_graph_num = 3

        for i in range(iter_graph_num):

            graph_name = ['Spatial Graph', 'Semantic Graph', 'Implicit Graph']
            graph_name = graph_name[i]

            graph_name = args.graph if iter_graph_num == 1 else graph_name

            # plotting
            fig = plt.figure(figsize=(10, 8), dpi=300)
            # hide axes
            plt.axis('off')
            plt.title('index: ' + str(idx) +'\n' + graph_name)
            plt.text(0, -0.05, output_text, fontsize=12, wrap=True)
            # plt.text(0.05, 0.06, 'Answer: ' + raw_answer, fontsize=12, wrap=True)
            # plt.text(0.05, 0.03, 'Prediction: ' + prediction, fontsize=12, wrap=True)





            ax1 = fig.add_subplot(1, 2, 2)
            ax1.imshow(img1)
            ax1.title.set_text('Main ' +str(study_id))
            ax2 = fig.add_subplot(1, 2, 1)
            ax2.imshow(img2)
            ax2.title.set_text('Reference '+ str(ref_id))

            ax1.axis('off')
            ax2.axis('off')

            if len(att_bef.shape) == 3:
                _, bef_indexes = torch.topk(att_bef[0].squeeze(), 3)
                _, aft_indexes = torch.topk(att_aft[0].squeeze(), 3)
            else:
                _, bef_indexes = torch.topk(att_bef[i][0][0].sum(1).sum(0), 3)
                _, aft_indexes = torch.topk(att_aft[i][0][0].sum(1).sum(0), 3)

            bef_centers = []
            bbx_labels = []
            for index in bef_indexes.flatten():
                if index >= 26:
                    index = index - 26
                ax1.add_patch(
                    patches.Rectangle(
                        (d_bb[index][0], d_bb[index][1]),
                        d_bb[index][2] - d_bb[index][0],
                        d_bb[index][3] - d_bb[index][1],
                        edgecolor='red',
                        fill=False
                    ))


                bbx_labels.append([float(d_bb[index][0]), float(d_bb[index][1]), id2cat[d_bb_label[int(index)]]])
                center = (d_bb[index][0] + d_bb[index][2]) / 2, (d_bb[index][1] + d_bb[index][3]) / 2
                bef_centers.append(center)
            bbx_labels = shift_bbx(bbx_labels)
            for item in bbx_labels:
                t = ax1.text(item[0], item[1], item[2], fontsize=9, wrap=True,
                         color='red')
                t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))
            # print ana dis corresponding bbx names
            bef_dis_names = []
            # for index in bef_indexes.flatten():
            #     # ana name
            #     # ana_name = id2cat[d_bb_label[int(index)]]
            #     if index >= 26:
            #         index = index - 26
            #     ana_name = id2cat[index]
            #     # disease name
            #     # index = index + 26
            #     # dis_id = d_bb_label[int(index)] -26
            #     # dis_name = id2dis[dis_id]
            #     # bef_dis_names.append(dis_name)
            #     print('before ane, dis name: %s, %s'% (ana_name, dis_name))
            # add line to connect all the centers
            # for i in range(len(bef_centers)):
            #     for j in range(i+1, len(bef_centers)):
            #         ax1.plot([bef_centers[i][0], bef_centers[j][0]], [bef_centers[i][1], bef_centers[j][1]], color='white', linewidth=1)

            bbx_labels = []
            for index in aft_indexes.flatten():

                if index >= 26:
                    index = index - 26
                ax2.add_patch(
                    patches.Rectangle(
                        (q_bb[index][0], q_bb[index][1]),
                        q_bb[index][2] - q_bb[index][0],
                        q_bb[index][3] - q_bb[index][1],
                        edgecolor='red',
                        fill=False,
                    ))
                bbx_labels.append([float(q_bb[index][0]), float(q_bb[index][1]), id2cat[q_bb_label[int(index)]]])
                # ax2.text(float(q_bb[index][0]), float(q_bb[index][1]), id2cat[int(index)], fontsize=9, wrap=True, color='red',backgroundcolor='white')

            bbx_labels = shift_bbx(bbx_labels)
            for item in bbx_labels:
                t = ax2.text(item[0], item[1], item[2], fontsize=9, wrap=True,
                         color='red', backgroundcolor='white')

                t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))
            # print ana dis corresponding bbx names
            # aft_dis_names = []
            # for index in aft_indexes.flatten():
            #     # ana name
            #     ana_name = id2cat[q_bb_label[int(index)]]
            #     # disease name
            #     index = index + 26
            #     dis_id = q_bb_label[int(index)] -26
            #     dis_name = id2dis[dis_id]
            #     aft_dis_names.append(dis_name)
            #     print('after ane, dis name: %s, %s'% (ana_name, dis_name))

            plt.show()
            print('a')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./configs/dynamic/dynamic_change_pos_mimic.yaml')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--entropy_weight', type=float, default=0.0)
    parser.add_argument('--visualize_every', type=int, default=10)
    parser.add_argument('--setting', type=str, default='mode2')
    parser.add_argument('--graph', type=str, default='all', choices=['implicit', 'semantic', 'spatial', 'all'])
    # parser.add_argument('--graph', type=str, default='implicit', choices=['implicit', 'semantic', 'spatial', 'all'])
    # parser.add_argument('--graph', type=str, default='semantic', choices=['implicit', 'semantic', 'spatial', 'all'])
    # parser.add_argument('--graph', type=str, default='all', choices=['implicit', 'semantic', 'spatial', 'all'])
    parser.add_argument('--use_wandb', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_fold', type=str, default='')
    parser.add_argument('--snapshot', type=int, default=5000)
    parser.add_argument('--feature_mode', type=str, default='location',
                        choices=['both', 'coords', 'location', 'single_ana',
                                 'single_loc'])  # both means ana+coords+location.
    parser.add_argument('--seed', type=int, default=1113)

    args = parser.parse_args()
    merge_cfg_from_file(args.cfg)
    cfg.data.feature_mode = args.feature_mode
    cfg.exp_name = args.setting + '_' + args.feature_mode
    cfg.train.graph = args.graph
    cfg.train.setting = args.setting

    if args.graph == 'semantic':
        args.resume_fold = 'mode2_location_semantic_0.0001_coef0.333000-0.333000_2022-11-12-20-57-32'
    elif args.graph == 'implicit':
        args.resume_fold = 'mode2_location_implicit_0.0001_coef0.333000-0.333000_2022-11-13-03-36-21'
    elif args.graph == 'spatial':
        args.resume_fold = 'mode2_location_spatial_0.0001_coef0.333000-0.333000_2022-11-13-00-15-25'
    elif args.graph == 'all':
        # args.resume_fold = 'mode2_location_all_0.0001_coef0.333000-0.333000_2022-11-16-09-10-29_1238'
        args.resume_fold = 'mode2_location_all_0.0001_coef0.400000-0.400000_2023-04-10-19-57-19_234'
    plotting_diff( checkpoint_num=18000)


# main file for plotting. not limited to diff question.
# this file is for plotting key bboxes of differene pairs