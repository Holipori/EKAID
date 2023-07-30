import os
import argparse
import json
import time
import sys

# sys.path.append('..')
print('current working path:', os.getcwd())
import numpy as np
import torch

torch.backends.cudnn.enabled = True
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from configs.config import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.modules import AddSpatialInfo, ChangeDetector
from models.dynamic_speaker_change_pos import DynamicSpeaker

from utils.utils import AverageMeter, accuracy, set_mode, load_checkpoint, \
    decode_sequence, coco_gen_format_save
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.mimic_utils import process_matrix
from pycocotools.coco import COCO
from evaluation import my_COCOEvalCap
import pexpect

import socket


def load_data(batch, extend=0):
    new_batch = [0 for i in range(len(batch))]
    new_batch[0] = torch.tensor(batch[0]).unsqueeze(0).clone().detach().float()
    new_batch[1] = torch.tensor(batch[1]).unsqueeze(0).clone().detach().float()
    new_batch[2] = torch.tensor(batch[2]).unsqueeze(0).clone().detach().float()
    new_batch[3] = torch.tensor(batch[3]).unsqueeze(0).clone().detach().float()
    new_batch[4] = torch.tensor(batch[4]).unsqueeze(0).clone().detach().float()
    new_batch[5] = torch.tensor(batch[5]).unsqueeze(0).clone().detach().float()
    new_batch[6] = torch.tensor(batch[6]).unsqueeze(0).clone().detach().float()
    new_batch[7] = torch.tensor(batch[7]).unsqueeze(0).clone().detach().float()
    new_batch[8] = torch.tensor(batch[8]).unsqueeze(0).clone().detach().float()
    new_batch[9] = torch.tensor(batch[9]).unsqueeze(0).clone().detach().float()
    new_batch[10] = torch.tensor(batch[10]).unsqueeze(0).clone().detach().float()
    new_batch[11] = torch.tensor(batch[11]).unsqueeze(0).clone().detach().float()
    new_batch[12] = torch.tensor(batch[12], dtype=torch.long).unsqueeze(0).clone().detach()
    if extend:
        new_batch[0] = new_batch[0].expand(extend, new_batch[0].shape[1], new_batch[0].shape[2])
        new_batch[1] = new_batch[1].expand(extend, new_batch[1].shape[1], new_batch[1].shape[2])
        new_batch[2] = new_batch[2].expand(extend, new_batch[2].shape[1], new_batch[2].shape[2])
        new_batch[3] = new_batch[3].expand(extend, new_batch[3].shape[1], new_batch[3].shape[2])
        new_batch[4] = new_batch[4].expand(extend, new_batch[4].shape[1], new_batch[4].shape[2])
        new_batch[5] = new_batch[5].expand(extend)
        new_batch[6] = new_batch[6].expand(extend, new_batch[6].shape[1], new_batch[6].shape[2])
        new_batch[7] = new_batch[7].expand(extend, new_batch[7].shape[1], new_batch[7].shape[2])
        new_batch[8] = new_batch[8].expand(extend, new_batch[8].shape[1], new_batch[8].shape[2])
        new_batch[9] = new_batch[9].expand(extend, new_batch[9].shape[1], new_batch[9].shape[2])
        new_batch[10] = new_batch[10].expand(extend, new_batch[10].shape[1], new_batch[10].shape[2])
        new_batch[11] = new_batch[11].expand(extend, new_batch[11].shape[1], new_batch[11].shape[2])
        new_batch[12] = new_batch[12].expand(extend, new_batch[12].shape[1])

    return new_batch


def question2id(raw_question, idx_to_word):
    word_to_idx = {v: k for k, v in idx_to_word.items()}
    output = [word_to_idx['<start>']]
    raw_question = raw_question.replace('?', ' ?').replace('.', ' .').replace(',', ' ,').replace('\'', ' \'').replace(
        '\"', ' \"').replace('(', ' (').replace(')', ' )').replace('-', ' -').replace('/', ' /').replace(':',
                                                                                                         ' :').replace(
        ';', ' ;')
    raw_question = raw_question.lower().split()
    for word in raw_question:
        if word in word_to_idx:
            output.append(word_to_idx[word])
        else:
            return 'Unexpected word error: {}'.format(word)
    for i in range(len(output), 20):
        output.append(0)
    return output


def example_questions():
    questions = '• What abnormalities are seen in this image?\n• What abnormalities are seen in the [location]?\n• Is there evidence of any abnormalities in this image?\n• Is this image normal?\n• Is there evidence of [abnormality] in this image?\n• Is there [abnormality]?\n• Is there [abnormality] in the [location]?\n• Which view is this image taken?\n• Is this PA view?\n• Is this AP view?\n• Where in the image is the [abnormality] located?\n• Where is the [abnormality]?\n• Is the [abnormality] located on the left side or right side?\n• Is the [abnormality] in the [location]?\n• What level is the [abnormality]?\n• What type is the [abnormality]?\n• What has changed compared to the reference image?\n• What has changed in the [location] area?\n'
    return questions


def run_vanilla():
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/dynamic/dynamic_change_pos_mimic.yaml')
    parser.add_argument('-p', '--resume_checkpoint', type=str,
                        default='/home/xinyue/SRDRL/experiments/final/mode2_location_all_0.0001_coef0.333000-0.333000_2023-04-11-12-55-40_1238/snapshots/checkpoint_34000.pt')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--feature_mode', type=str, default='location',
                        choices=['both', 'coords', 'location', 'single_ana', 'single_loc'])
    args = parser.parse_args()
    merge_cfg_from_file(args.cfg)
    # assert cfg.exp_name == os.path.basename(args.cfg).replace('.yaml', '')

    # Device configuration
    use_cuda = torch.cuda.is_available()
    if args.gpu == -1:
        gpu_ids = cfg.gpu_id
    else:
        gpu_ids = [args.gpu]
    torch.backends.cudnn.enabled = True
    default_gpu_device = gpu_ids[0]
    torch.cuda.set_device(default_gpu_device)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Experiment configuration
    exp_dir = cfg.exp_dir
    exp_name = 'mimic-diff'

    img_dir = '/home/xinyue/dataset/mimic-cxr-png'
    df_path = 'data/mimic_pair_questions.csv'
    df = pd.read_csv(df_path)
    df_all_path = 'data/mimic_all.csv'
    df_all = pd.read_csv(df_all_path)

    checkpoint = load_checkpoint(args.resume_checkpoint)
    change_detector_state = checkpoint['change_detector_state']
    speaker_state = checkpoint['speaker_state']
    annotation_file = 'data/mimic_gt_captions_test.json'
    # Data loading part
    train_dataset, train_loader = create_dataset(cfg, 'train')
    idx_to_word = train_dataset.get_idx_to_word()
    test_dataset, test_loader = create_dataset(cfg, 'all')

    # Load modules
    # change_detector = ChangeDetectorDoubleAttDyn(cfg)
    change_detector = ChangeDetector(cfg, train_dataset.word_to_idx)
    change_detector.load_state_dict(change_detector_state)
    change_detector = change_detector.to(device)

    speaker = DynamicSpeaker(cfg, len(train_dataset.get_idx_to_word()) + 1)
    speaker.load_state_dict(speaker_state)
    speaker.to(device)

    spatial_info = AddSpatialInfo()
    spatial_info.to(device)

    print(change_detector)
    print(speaker)
    print(spatial_info)

    set_mode('eval', [change_detector, speaker])
    with torch.no_grad():
        test_iter_start_time = time.time()

        predictions = {}
        result_sents_neg = {}
        random_idx = np.random.randint(0, len(test_dataset))
        batch = load_data(test_dataset[random_idx], 64)
        study_id = df.iloc[random_idx]['study_id']
        ref_id = df.iloc[random_idx]['ref_id']
        main_dicom_id = df_all[df_all['study_id'] == study_id]['dicom_id'].values[0]
        ref_dicom_id = df_all[df_all['study_id'] == ref_id]['dicom_id'].values[0]

        main_img_path = os.path.join(img_dir, main_dicom_id + '.png')
        ref_img_path = os.path.join(img_dir, ref_dicom_id + '.png')
        plt.figure()
        plt.imshow(plt.imread(main_img_path), cmap='gray')
        plt.title('Main image (current)')
        plt.show()
        plt.figure()
        plt.imshow(plt.imread(ref_img_path), cmap='gray')
        plt.title('Reference image')
        plt.show()

        d_feats, sc_feats, labels, sc_pos_labels, masks, pair_index, d_adj_matrix, q_adj_matrix, d_sem_adj_matrix, q_sem_dj_matrix, d_bb, q_bb, question = batch

        # ask for input
        while 1:
            raw_question = input('Please input a question:')
            question = question2id(raw_question, idx_to_word)
            if 'error' in question:
                print(question)
                print('example questions:\n')
                print(example_questions())
                continue
            else:
                question = torch.tensor(question).unsqueeze(0).expand(64, 20)
                break

        batch_size = d_feats.size(0)

        d_feats, sc_feats = d_feats.to(device), sc_feats.to(device)
        question = question.to(device)
        sc_pos_labels = sc_pos_labels.to(device)
        # d_feats, nsc_feats, sc_feats = \
        #     spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)
        labels, masks = labels.to(device), masks.to(device)

        d_adj_matrix = process_matrix(d_adj_matrix, cfg, d_feats.shape[1], d_feats.device, type='spatial')
        q_adj_matrix = process_matrix(q_adj_matrix, cfg, sc_feats.shape[1], sc_feats.device, type='spatial')

        d_sem_adj_matrix = process_matrix(d_sem_adj_matrix, cfg, d_feats.shape[1], d_feats.device, type='semantic')
        q_sem_dj_matrix = process_matrix(q_sem_dj_matrix, cfg, sc_feats.shape[1], sc_feats.device, type='semantic')

        chg_pos_logits, chg_pos_att_bef, chg_pos_att_aft, \
        chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff = change_detector(d_feats, sc_feats, d_adj_matrix,
                                                                                q_adj_matrix, d_sem_adj_matrix,
                                                                                q_sem_dj_matrix, d_bb, q_bb, question,
                                                                                setting=cfg.train.setting)

        speaker_output_pos, _ = speaker._sample(chg_pos_feat_bef,
                                                chg_pos_feat_aft,
                                                chg_pos_feat_diff,
                                                labels, cfg, sample_max=1)

        answer = decode_sequence(idx_to_word, speaker_output_pos)
        raw_question = decode_sequence(idx_to_word, question)
        print('question:', raw_question[0])
        print(answer[0])


def get_patient_id(random_idx, df, df_all):
    subject_id = df.iloc[random_idx]['subject_id']
    study_id = df.iloc[random_idx]['study_id']
    ref_id = df.iloc[random_idx]['ref_id']
    main_dicom_id = df_all[df_all['study_id'] == study_id]['dicom_id'].values[0]
    ref_dicom_id = df_all[df_all['study_id'] == ref_id]['dicom_id'].values[0]
    return subject_id, study_id, ref_id, main_dicom_id, ref_dicom_id


def run(random_idx, question, test_dataset, df, df_all, img_dir, idx_to_word, change_detector, speaker, device):
    with torch.no_grad():
        test_iter_start_time = time.time()

        predictions = {}
        result_sents_neg = {}
        batch = load_data(test_dataset[random_idx], 64)
        study_id = df.iloc[random_idx]['study_id']
        ref_id = df.iloc[random_idx]['ref_id']
        main_dicom_id = df_all[df_all['study_id'] == study_id]['dicom_id'].values[0]
        ref_dicom_id = df_all[df_all['study_id'] == ref_id]['dicom_id'].values[0]

        main_img_path = os.path.join(img_dir, main_dicom_id + '.png')
        ref_img_path = os.path.join(img_dir, ref_dicom_id + '.png')
        plt.figure()
        plt.imshow(plt.imread(main_img_path), cmap='gray')
        plt.title('Main image (current)')
        plt.show()
        plt.figure()
        plt.imshow(plt.imread(ref_img_path), cmap='gray')
        plt.title('Reference image')
        plt.show()

        d_feats, sc_feats, labels, sc_pos_labels, masks, pair_index, d_adj_matrix, q_adj_matrix, d_sem_adj_matrix, q_sem_dj_matrix, d_bb, q_bb, _ = batch

        batch_size = d_feats.size(0)

        d_feats, sc_feats = d_feats.to(device), sc_feats.to(device)
        question = question.to(device)
        sc_pos_labels = sc_pos_labels.to(device)
        # d_feats, nsc_feats, sc_feats = \
        #     spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)
        labels, masks = labels.to(device), masks.to(device)

        d_adj_matrix = process_matrix(d_adj_matrix, cfg, d_feats.shape[1], d_feats.device, type='spatial')
        q_adj_matrix = process_matrix(q_adj_matrix, cfg, sc_feats.shape[1], sc_feats.device, type='spatial')

        d_sem_adj_matrix = process_matrix(d_sem_adj_matrix, cfg, d_feats.shape[1], d_feats.device, type='semantic')
        q_sem_dj_matrix = process_matrix(q_sem_dj_matrix, cfg, sc_feats.shape[1], sc_feats.device, type='semantic')

        chg_pos_logits, chg_pos_att_bef, chg_pos_att_aft, \
        chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff = change_detector(d_feats, sc_feats, d_adj_matrix,
                                                                                q_adj_matrix, d_sem_adj_matrix,
                                                                                q_sem_dj_matrix, d_bb, q_bb, question,
                                                                                setting=cfg.train.setting)

        speaker_output_pos, _ = speaker._sample(chg_pos_feat_bef,
                                                chg_pos_feat_aft,
                                                chg_pos_feat_diff,
                                                labels, cfg, sample_max=1)

        answer = decode_sequence(idx_to_word, speaker_output_pos)
        raw_question = decode_sequence(idx_to_word, question)
        print('question:', raw_question[0])
        print(answer[0])
        return answer[0]


def send_file(c, file_path):
    # Open the file in binary mode
    f = open(file_path, 'rb')
    file_name = os.path.basename(file_path)
    print('Sending', file_name, 'to the client...')
    # Get the file size in bytes
    file_size = os.path.getsize(file_path)
    file_header = str(file_size) + ',' + file_name
    # Send the file size to the client
    c.send(file_header.encode())
    conformation = c.recv(1024)
    assert conformation.decode() == 'T'
    # Read the file and send it to the client
    data = f.read()
    c.send(data)
    conformation = c.recv(1024)
    assert conformation.decode() == 'T'
    # Close the file
    f.close()
    print('Done sending', file_name, 'to the client')


def main():
    ## initiate model
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/dynamic/dynamic_change_pos_mimic.yaml')
    parser.add_argument('-c', '--resume_checkpoint', type=str,
                        default='/home/xinyue/SRDRL/experiments/final/mode2_location_all_0.0001_coef0.333000-0.333000_2023-04-11-12-55-40_1238/snapshots/checkpoint_34000.pt')
    parser.add_argument('-p', '--mimic_png_path', type=str, default='/home/xinyue/dataset/mimic-cxr-png')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--feature_mode', type=str, default='location',
                        choices=['both', 'coords', 'location', 'single_ana', 'single_loc'])
    args = parser.parse_args()
    merge_cfg_from_file(args.cfg)
    # assert cfg.exp_name == os.path.basename(args.cfg).replace('.yaml', '')

    # Device configuration
    use_cuda = torch.cuda.is_available()
    if args.gpu == -1:
        gpu_ids = cfg.gpu_id
    else:
        gpu_ids = [args.gpu]
    torch.backends.cudnn.enabled = True
    default_gpu_device = gpu_ids[0]
    torch.cuda.set_device(default_gpu_device)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Experiment configuration
    exp_dir = cfg.exp_dir
    exp_name = 'mimic-diff'

    img_dir = args.mimic_png_path
    df_path = 'data/mimic_pair_questions.csv'
    df = pd.read_csv(df_path)
    df_all_path = 'data/mimic_all.csv'
    df_all = pd.read_csv(df_all_path)

    checkpoint = load_checkpoint(args.resume_checkpoint)
    change_detector_state = checkpoint['change_detector_state']
    speaker_state = checkpoint['speaker_state']
    annotation_file = 'data/mimic_gt_captions_test.json'
    # Data loading part
    train_dataset, train_loader = create_dataset(cfg, 'train')
    idx_to_word = train_dataset.get_idx_to_word()
    test_dataset, test_loader = create_dataset(cfg, 'all')

    # Load modules
    # change_detector = ChangeDetectorDoubleAttDyn(cfg)
    change_detector = ChangeDetector(cfg, train_dataset.word_to_idx)
    change_detector.load_state_dict(change_detector_state)
    change_detector = change_detector.to(device)

    speaker = DynamicSpeaker(cfg, len(train_dataset.get_idx_to_word()) + 1)
    speaker.load_state_dict(speaker_state)
    speaker.to(device)

    spatial_info = AddSpatialInfo()
    spatial_info.to(device)

    print(change_detector)
    print(speaker)
    print(spatial_info)

    set_mode('eval', [change_detector, speaker])

    ### Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Define the port on which you want to connect
    port = 4000
    # Bind to the port
    s.bind(('0.0.0.0', port))
    # Put the socket into listening mode
    s.listen(5)
    # A loop to accept requests and send files until exit()
    while True:
        print('started listening...')
        # Establish connection with client.
        c, addr = s.accept()
        print('Got connection from', addr)
        random_idx = np.random.randint(0, len(test_dataset))
        print('random_idx:', random_idx)
        while True:

            # Receive the file name from the client
            command = c.recv(1024).decode()

            # unexpected exit
            if command == '':
                print('lost connection with', addr)
                c.close()
                break
            # normal 'exit()'
            elif command == 'exit()':
                # Break the loop and close the connection
                print('Exiting the server')
                c.close()
                break
            elif command == 'question':
                c.send('ready'.encode())  # send a signal to the client to start sending question
                raw_question = c.recv(1024).decode()
                question = question2id(raw_question, idx_to_word)
                if 'error' in question:
                    print(question)
                    print('example questions:\n')
                    print(example_questions())
                    c.send('error'.encode())
                    confirmation = c.recv(1024).decode()
                    assert confirmation == 'T'
                    c.send(example_questions().encode())
                    continue
                question = torch.tensor(question).unsqueeze(0).expand(64, 20)
                answer = run(random_idx, question, test_dataset, df, df_all, img_dir, idx_to_word, change_detector,
                             speaker, device)
                c.send(answer.encode())
                continue
            elif command == 'refresh':
                random_idx = np.random.randint(0, len(test_dataset))
                print('refreshed')
                print('random_idx:', random_idx)
                continue
            elif command == 'load_image':
                subject_id, study_id, ref_id, main_dicom_id, ref_dicom_id = get_patient_id(random_idx, df, df_all)
                main_img_path = os.path.join(img_dir, main_dicom_id + '.png')
                ref_img_path = os.path.join(img_dir, ref_dicom_id + '.png')
                send_file(c, main_img_path)
                send_file(c, ref_img_path)
                c.send(str(subject_id).encode())
                continue



        # Close the connection with the client
        c.close()


if __name__ == "__main__":
    main()







