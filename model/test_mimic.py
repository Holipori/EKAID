import os
import argparse
import json
import time
import numpy as np
import torch
torch.backends.cudnn.enabled  = True
import torch.nn as nn
import torch.nn.functional as F

from configs.config import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.modules import AddSpatialInfo, ChangeDetector
from models.dynamic_speaker_change_pos import DynamicSpeaker

from utils.utils import AverageMeter, accuracy, set_mode, load_checkpoint, \
                        decode_sequence, coco_gen_format_save
from tqdm import tqdm

from utils.mimic_utils import process_matrix
from pycocotools.coco import COCO
from evaluation import my_COCOEvalCap

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='configs/dynamic/dynamic_change_pos_mimic.yaml')
parser.add_argument('-p','--resume_checkpoint', type=str, default='./experiments/final/mode2_location_all_0.0001_coef0.333000-0.333000_2023-04-11-12-55-40_1238/snapshots/checkpoint_34000.pt')
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--feature_mode', type=str, default='location', choices= ['both','coords', 'location', 'single_ana', 'single_loc'])
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

output_dir = os.path.join(exp_dir, exp_name)

test_output_dir = os.path.join(output_dir, 'test_output')
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)
caption_output_path = os.path.join(test_output_dir, 'captions', 'test')
if not os.path.exists(caption_output_path):
    os.makedirs(caption_output_path)
att_output_path = os.path.join(test_output_dir, 'attentions', 'test')
if not os.path.exists(att_output_path):
    os.makedirs(att_output_path)


checkpoint = load_checkpoint(args.resume_checkpoint)
change_detector_state = checkpoint['change_detector_state']
speaker_state = checkpoint['speaker_state']
annotation_file = 'data/mimic_gt_captions_test.json'
# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
idx_to_word = train_dataset.get_idx_to_word()
test_dataset, test_loader = create_dataset(cfg, 'test')

# Load modules
# change_detector = ChangeDetectorDoubleAttDyn(cfg)
change_detector = ChangeDetector(cfg, train_dataset.word_to_idx)
change_detector.load_state_dict(change_detector_state)
change_detector = change_detector.to(device)

speaker = DynamicSpeaker(cfg, len(train_dataset.get_idx_to_word())+1)
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
    for i, batch in enumerate(tqdm(test_loader)):

        d_feats, sc_feats, labels, sc_pos_labels,  masks, pair_index, d_adj_matrix, q_adj_matrix, d_sem_adj_matrix, q_sem_dj_matrix,d_bb,q_bb, question= batch

        batch_size = d_feats.size(0)

        d_feats,  sc_feats = d_feats.to(device), sc_feats.to(device)
        question = question.to(device)
        sc_pos_labels= sc_pos_labels.to(device)
        # d_feats, nsc_feats, sc_feats = \
        #     spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)
        labels, masks = labels.to(device), masks.to(device)

        d_adj_matrix = process_matrix(d_adj_matrix,cfg, d_feats.shape[1], d_feats.device, type = 'spatial')
        q_adj_matrix = process_matrix(q_adj_matrix,cfg, sc_feats.shape[1], sc_feats.device, type = 'spatial')

        d_sem_adj_matrix = process_matrix(d_sem_adj_matrix, cfg, d_feats.shape[1], d_feats.device, type='semantic')
        q_sem_dj_matrix = process_matrix(q_sem_dj_matrix, cfg, sc_feats.shape[1], sc_feats.device, type='semantic')

        chg_pos_logits, chg_pos_att_bef, chg_pos_att_aft, \
        chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff = change_detector(d_feats, sc_feats, d_adj_matrix, q_adj_matrix,d_sem_adj_matrix, q_sem_dj_matrix,d_bb,q_bb, question, setting = cfg.train.setting)

        speaker_output_pos, _ = speaker._sample(chg_pos_feat_bef,
                                                chg_pos_feat_aft,
                                                chg_pos_feat_diff,
                                                labels, cfg, sample_max=1)

        gen_sents_pos = decode_sequence(idx_to_word, speaker_output_pos)

        chg_pos_att_bef = chg_pos_att_bef.cpu().numpy()
        chg_pos_att_aft = chg_pos_att_aft.cpu().numpy()

        dummy = np.ones_like(chg_pos_att_bef)
        for j in range(batch_size):
            predict_sent = gen_sents_pos[j]
            predictions[str(int(pair_index[j]))] = predict_sent



    test_iter_end_time = time.time() - test_iter_start_time
    print('Test took %.4f seconds' % test_iter_end_time)

    result_save_path_pos = os.path.join(caption_output_path, 'test_results_%s.json'%args.feature_mode)
    print(result_save_path_pos)
    coco_gen_format_save(predictions, result_save_path_pos)

    coco = COCO(annotation_file)
    coco_result = coco.loadRes(result_save_path_pos)
    coco_eval = my_COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()
    # print output evaluation scores
    output = []
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
        output.append(score)


