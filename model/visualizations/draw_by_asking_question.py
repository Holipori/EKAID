import os
import argparse
import time
import numpy as np
import torch
import pandas as pd
import nltk

from configs.config import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.modules import ChangeDetector, AddSpatialInfo
from models.dynamic_speaker_change_pos import DynamicSpeaker
from utils.logger import Logger
from utils.utils import AverageMeter, accuracy, set_mode, load_checkpoint, \
                        decode_sequence, build_optimizer, coco_gen_format_save, \
                        one_hot_encode, EntropyLoss
from utils.mimic_utils import process_matrix
from draw_single import plot_by_id

def load_data(data_batch, question):
    data_batch = list(data_batch)
    data_batch[-1] = question
    output = []
    for d in data_batch:
        d = torch.tensor(d).unsqueeze(0)
        # expand the first dimension to 64
        target_shape = [64] + list(d.shape[1:])
        d = d.expand(target_shape)
        output.append(d)
    return tuple(output)

def find_feat_idx(study_id, splits):
    path = os.path.join('visualizations', '../data/datasets/mimic_pair_questions.csv')
    df = pd.read_csv(path)
    # find the index of the study_id
    idx = df[df['study_id'] == study_id].index[0]

    if idx in splits['train']:
        return idx
    elif idx in splits['val']:
        return idx - len(splits['train'])
    elif idx in splits['test']:
        return idx - len(splits['train']) - len(splits['val'])

def question_process(raw_question, vocab, max_seq_length):
    question_list = nltk.word_tokenize(raw_question.lower())
    question_list = [int(vocab[word]) for word in question_list]
    question = np.zeros(max_seq_length, dtype=np.int64)
    question[:len(question_list)] = np.array(question_list)
    return question


def main(study_id, question):
    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/dynamic/dynamic_change_pos_mimic.yaml')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--entropy_weight', type=float, default=0.0)
    parser.add_argument('--visualize_every', type=int, default=10)
    parser.add_argument('--setting', type=str, default='mode2')
    parser.add_argument('--graph', type=str, default='all', choices=['implicit', 'semantic', 'spatial', 'all', 'i+s'])
    parser.add_argument('--use_wandb', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--resume_fold', type=str, default='experiments/final/mode2_location_all_0.0001_coef0.333000-0.333000_2022-11-16-09-10-29_1238')
    parser.add_argument('--snapshot', type=int, default=22000)
    parser.add_argument('--feature_mode', type=str, default='location', choices= ['both','coords', 'location', 'single_ana', 'single_loc']) # both means ana+coords+location.
    parser.add_argument('--seed', type=int, default=1113)
    parser.add_argument('--coef_sem', type=float, default=0.333)
    parser.add_argument('--coef_spa', type=float, default=0.333)

    args = parser.parse_args()
    merge_cfg_from_file(args.cfg)
    cfg.model.coef_sem = args.coef_sem
    cfg.model.coef_spa = args.coef_spa
    # cfg.data.test.batch_size = 1

    cfg.data.feature_mode = args.feature_mode
    cfg.exp_name = args.setting + '_' + args.feature_mode
    cfg.train.graph = args.graph
    if args.setting == 'mode2':
        cfg.exp_name += '_{}'.format(args.graph)
    exp_name = cfg.exp_name
    exp_name = exp_name+'_'+str(cfg.train.optim.lr) + '_' + 'coef%f-%f'%(args.coef_sem,args.coef_spa) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '_' + str(args.seed)
    print('exp_name',exp_name)


    # Create dataset
    test_dataset, test_loader = create_dataset(cfg, 'test')
    idx_to_word = test_dataset.get_idx_to_word()

    # process query
    idx = find_feat_idx(study_id, test_dataset.splits)
    question = question_process(question, test_dataset.word_to_idx, 20)

    device = 'cuda'
    # Load pre-trained model
    print('loading checkpoints')
    snapshot_dir = os.path.join(args.resume_fold, 'snapshots')
    snapshot_file = 'checkpoint_%d.pt' % args.snapshot
    snapshot_full_path = os.path.join(snapshot_dir, snapshot_file)
    checkpoint = load_checkpoint(snapshot_full_path)
    change_detector_state = checkpoint['change_detector_state']
    speaker_state = checkpoint['speaker_state']

    # Load modules
    change_detector = ChangeDetector(cfg, test_dataset.word_to_idx)
    change_detector.load_state_dict(change_detector_state)
    change_detector = change_detector.to(device)

    speaker = DynamicSpeaker(cfg, len(test_dataset.get_idx_to_word()) + 1)
    speaker.load_state_dict(speaker_state)
    speaker.to(device)
    print('checkpoints loading successfully')

    # Set model to evaluation mode
    set_mode('eval', [change_detector, speaker])


    # Evaluate model on test set
    total_acc = 0
    total_len = 0
    for i, ldata in enumerate(test_loader):
        if i > 0:
            break
    data = load_data(test_dataset[idx], question)
    # data = ldata
    d_feats, sc_feats, labels, sc_pos_labels,  masks, pair_index, d_adj_matrix, q_adj_matrix, d_sem_adj_matrix, q_sem_dj_matrix,d_bb,q_bb, question = data

    d_feats, sc_feats = d_feats.to(device), sc_feats.to(device)
    question = question.to(device)
    sc_pos_labels = sc_pos_labels.to(device)
    labels, masks = labels.to(device), masks.to(device)

    d_adj_matrix = process_matrix(d_adj_matrix, cfg, d_feats.shape[1], d_feats.device, type='spatial')
    q_adj_matrix = process_matrix(q_adj_matrix, cfg, sc_feats.shape[1], sc_feats.device, type='spatial')
    d_sem_adj_matrix = process_matrix(d_sem_adj_matrix, cfg, d_feats.shape[1], d_feats.device, type='semantic')
    q_sem_dj_matrix = process_matrix(q_sem_dj_matrix, cfg, sc_feats.shape[1], sc_feats.device, type='semantic')
    # Forward pass
    with torch.no_grad():
        chg_pos_logits, chg_pos_att_bef, chg_pos_att_aft, chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff \
            = change_detector(d_feats, sc_feats, d_adj_matrix, q_adj_matrix, d_sem_adj_matrix, q_sem_dj_matrix, d_bb, q_bb, question, setting=cfg.train.setting, graph=args.graph)
        speaker_output_pos, _ = speaker._sample(chg_pos_feat_bef,
                                                chg_pos_feat_aft,
                                                chg_pos_feat_diff,
                                                labels, cfg, sample_max=0)

    gen_sents_pos = decode_sequence(idx_to_word, speaker_output_pos)

    # Compute accuracy
    ans_count = {}
    for test_j in range(speaker_output_pos.size(0)):
        gts = decode_sequence(idx_to_word, labels[test_j][:, 1:])
        sent_pos = gen_sents_pos[test_j]
        ans_count[sent_pos] = ans_count.get(sent_pos, 0) + 1
        # print('gt:', gts)
        print('pred:', sent_pos)
        print('-----------------')
        # break

    # sort ans_count
    ans_count = sorted(ans_count.items(), key=lambda x: x[1], reverse=True)
    print('ans_count', ans_count)



if __name__ == '__main__':
    question = 'where is the lung opacity?'
    study_id = 51765127
    main(study_id, question)
    plot_by_id(study_id)
