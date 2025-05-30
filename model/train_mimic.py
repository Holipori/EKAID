import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import json
import argparse
import time
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
import random
# import h5py
import wandb

from configs.config import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.modules import ChangeDetector, AddSpatialInfo
from models.dynamic_speaker_change_pos import DynamicSpeaker
from utils.logger import Logger
from utils.utils import AverageMeter, accuracy, set_mode, save_checkpoint, \
                        LanguageModelCriterion, decode_sequence, decode_beams, \
                        build_optimizer, coco_gen_format_save, one_hot_encode, \
                        EntropyLoss,load_checkpoint
from utils.mimic_utils import process_matrix
from tqdm import tqdm
from pycocotools.coco import COCO
from evaluation import my_COCOEvalCap

# os.environ['CUDA_LAUNCH_BLOCKING']='1'
# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='configs/dynamic/dynamic_change_pos_mimic.yaml')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--entropy_weight', type=float, default=0.0)
parser.add_argument('--visualize_every', type=int, default=10)
parser.add_argument('--setting', type=str, default='mode2', help='mode0: use image directly; mode2: use region features')
parser.add_argument('--graph', type=str, default='all', choices=['implicit', 'semantic', 'spatial', 'all', 'i+s'])
parser.add_argument('--use_wandb', type=bool, default=False)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--resume_fold', type=str, default='mode2_location_all_0.0001_2022-09-16-22-43-34')
parser.add_argument('--snapshot', type=int, default=22000)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--feature_mode', type=str, default='both', choices= ['both', 'location', 'single_ana', 'single_loc'])
parser.add_argument('--eval_target', type=str, default='test', choices=['test', 'val'])
parser.add_argument('--seed', type=int, default=1238)
parser.add_argument('--coef_sem', type=float, default=0.333)
parser.add_argument('--coef_spa', type=float, default=0.333)
parser.add_argument('--lr', type=float, default=0.0001)

args = parser.parse_args()
merge_cfg_from_file(args.cfg)
cfg.model.coef_sem = args.coef_sem
cfg.model.coef_spa = args.coef_spa
cfg.train.optim.lr = args.lr
cfg.data.feature_mode = args.feature_mode
cfg.exp_name = args.setting + '_' + args.feature_mode
cfg.train.graph = args.graph
if args.setting == 'mode2':
    cfg.exp_name += '_{}'.format(args.graph)
exp_name = cfg.exp_name
if cfg.data.train.empty_image:
    exp_name = cfg.exp_name + '_empty'
exp_name = exp_name+'_'+str(cfg.train.optim.lr) + '_' + 'coef%f-%f'%(args.coef_sem,args.coef_spa) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '_' + str(args.seed)
print('Run name:',exp_name)
if args.use_wandb:
    name = args.setting
    wandb.init(project='image difference', config=args, name=exp_name, allow_val_change=True)
    # wandb.config.update(args, )
    args = wandb.config


cfg.train.setting = args.setting
print('mode', args.feature_mode)

# Device configuration
use_cuda = torch.cuda.is_available()
gpu_ids = cfg.gpu_id
# torch.backends.cudnn.enabled = False
default_gpu_device = gpu_ids[0]
torch.cuda.set_device(default_gpu_device)
device = torch.device("cuda" if use_cuda else "cpu")

# Experiment configuration
exp_dir = cfg.exp_dir
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

output_dir = os.path.join(exp_dir, 'temp', exp_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cfg_file_save = os.path.join(output_dir, 'cfg.json')
json.dump(cfg, open(cfg_file_save, 'w'))

sample_dir = os.path.join(output_dir, 'eval_gen_samples')
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
sample_subdir_format = '%s_samples_%d'

sent_dir = os.path.join(output_dir, 'eval_sents')
if not os.path.exists(sent_dir):
    os.makedirs(sent_dir)
sent_subdir_format = '%s_sents_%d'

snapshot_dir = os.path.join(output_dir, 'snapshots')
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)
snapshot_file_format = 'checkpoint_%d.pt'


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
val_dataset, val_loader = create_dataset(cfg, 'test' if args.eval_target == 'test' else 'val')
annotation_file = 'data/mimic_gt_captions_test.json' if args.eval_target == 'test' else 'data/mimic_gt_captions_val.json'
train_size = len(train_dataset)
val_size = len(val_dataset)



resume = args.resume
if resume:
    print('loading checkpoints')
    snapshot_full_path = args.checkpoint
    assert os.path.exists(snapshot_full_path)
    checkpoint = load_checkpoint(snapshot_full_path)
    change_detector_state = checkpoint['change_detector_state']
    speaker_state = checkpoint['speaker_state']

    # Load modules
    change_detector = ChangeDetector(cfg, train_dataset.word_to_idx)
    change_detector.load_state_dict(change_detector_state)
    change_detector = change_detector.to(device)

    speaker = DynamicSpeaker(cfg, len(train_dataset.get_idx_to_word())+1)
    speaker.load_state_dict(speaker_state)
    speaker.to(device)
    print('checkpoints loading successfully')
else:
    # Create model
    change_detector = ChangeDetector(cfg, train_dataset.word_to_idx)
    change_detector.to(device)

    speaker = DynamicSpeaker(cfg, len(train_dataset.get_idx_to_word())+1)
    speaker.to(device)

if args.use_wandb:
    wandb.watch([change_detector,speaker])


spatial_info = AddSpatialInfo()
spatial_info.to(device)

print(change_detector)
print(speaker)
print(spatial_info)

with open(os.path.join(output_dir, 'model_print'), 'w') as f:
    print(change_detector, file=f)
    print(speaker, file=f)
    print(spatial_info, file=f)



# Define loss function and optimizer
lang_criterion = LanguageModelCriterion().to(device)
# entropy_criterion = EntropyLoss().to(device)
all_params = list(change_detector.parameters()) + list(speaker.parameters())
optimizer = build_optimizer(all_params, cfg)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cfg.train.optim.step_size,
    gamma=cfg.train.optim.gamma)

# Train loop
t = 0
epoch = 0
best = 0

set_mode('train', [change_detector, speaker])
ss_prob = speaker.ss_prob

while t < cfg.train.max_iter:
# while epoch < cfg.train.max_epoch:
    print('Starting epoch %d' % epoch)
    lr_scheduler.step()
    print(lr_scheduler.optimizer.defaults['lr'])
    speaker_loss_avg = AverageMeter()
    speaker_pos_loss_avg = AverageMeter()
    total_loss_avg = AverageMeter()
    if epoch > cfg.train.scheduled_sampling_start and cfg.train.scheduled_sampling_start >= 0: # skip
        frac = (epoch - cfg.train.scheduled_sampling_start) // cfg.train.scheduled_sampling_increase_every
        ss_prob_prev = ss_prob
        ss_prob = min(cfg.train.scheduled_sampling_increase_prob * frac,
                      cfg.train.scheduled_sampling_max_prob)
        speaker.ss_prob = ss_prob
        if ss_prob_prev != ss_prob:
            print('Updating scheduled sampling rate: %.4f -> %.4f' % (ss_prob_prev, ss_prob))
    for i, batch in enumerate(train_loader):
        iter_start_time = time.time()

        d_feats, sc_feats, labels, sc_pos_labels, masks, pair_index, d_adj_matrix, q_adj_matrix, d_sem_adj_matrix, q_sem_dj_matrix, d_bb, q_bb, question  = batch

        batch_size = d_feats.size(0)
        labels = labels.squeeze(1)
        sc_pos_labels = sc_pos_labels.squeeze(1)
        masks = masks.squeeze(1).float()

        d_feats, sc_feats = d_feats.to(device), sc_feats.to(device)
        question = question.to(device)
        sc_pos_labels = sc_pos_labels.to(device)
        labels, masks = labels.to(device), masks.to(device)
        d_adj_matrix, q_adj_matrix = d_adj_matrix.to(device), q_adj_matrix.to(device)
        d_sem_adj_matrix, q_sem_dj_matrix = d_sem_adj_matrix.to(device), q_sem_dj_matrix.to(device)

        optimizer.zero_grad()

        if args.setting == 'mode2':
            d_adj_matrix = process_matrix(d_adj_matrix,cfg, d_feats.shape[1], d_feats.device, type = 'spatial')
            q_adj_matrix = process_matrix(q_adj_matrix,cfg, sc_feats.shape[1], sc_feats.device, type = 'spatial')

            d_sem_adj_matrix = process_matrix(d_sem_adj_matrix, cfg, d_feats.shape[1], d_feats.device, type = 'semantic')
            q_sem_adj_matrix = process_matrix(q_sem_dj_matrix, cfg, sc_feats.shape[1], sc_feats.device, type = 'semantic')

        chg_pos_logits, chg_pos_att_bef, chg_pos_att_aft, \
        chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff = change_detector(d_feats, sc_feats, d_adj_matrix, q_adj_matrix,d_sem_adj_matrix, q_sem_adj_matrix,d_bb,q_bb, question, setting = cfg.train.setting, graph=args.graph)
        # chg_neg_logits, chg_neg_att_bef, chg_neg_att_aft, \
        # chg_neg_feat_bef, chg_neg_feat_aft, chg_neg_feat_diff = change_detector(d_feats, nsc_feats)


        speaker_output_pos, speaker_output_pos_tag = speaker._forward(chg_pos_feat_bef,
                                              chg_pos_feat_aft,
                                              chg_pos_feat_diff,
                                              labels)
        dynamic_atts = speaker.get_module_weights() # (batch, seq_len, 3)


        speaker_loss =  lang_criterion(speaker_output_pos, labels[:,1:], masks[:,1:])
        speaker_loss_val = speaker_loss.item()

        # entropy_loss = -args.entropy_weight * entropy_criterion(dynamic_atts, masks[:,1:])
        att_sum = (chg_pos_att_bef.sum() + chg_pos_att_aft.sum()) / (2 * batch_size)
        total_loss = speaker_loss  + 2.5e-03 * att_sum
        total_loss_val = total_loss.item()

        speaker_loss_avg.update(speaker_loss_val, 2 * batch_size)
        # speaker_pos_loss_avg.update(speaker_pos_loss_val, 2 * batch_size)
        total_loss_avg.update(total_loss_val, 2 * batch_size)

        stats = {}
        # stats['entropy_loss'] = entropy_loss.item()
        stats['speaker_loss'] = speaker_loss_val
        stats['avg_speaker_loss'] = speaker_loss_avg.avg
        stats['total_loss'] = total_loss_val
        stats['avg_total_loss'] = total_loss_avg.avg
        if args.use_wandb:
            wandb.log({'speaker_loss':speaker_loss_val,
                       'avg_speaker_loss':speaker_loss_avg.avg,
                       'total_loss':total_loss_val,
                       'avg_total_loss':total_loss_avg.avg,
                       })

        #results, sample_logprobs = model(d_feats, q_feats, labels, cfg=cfg, mode='sample')
        total_loss.backward()
        optimizer.step()

        iter_end_time = time.time() - iter_start_time

        t += 1

        if t % cfg.train.log_interval == 0:
            print(epoch, i, t, stats, iter_end_time)
            print(epoch, float(i * batch_size) / train_size, stats, 'loss')

        # Evaluation
        if t % cfg.train.snapshot_interval == 0:
            speaker_state = speaker.state_dict()
            chg_det_state = change_detector.state_dict()
            checkpoint = {
                'change_detector_state': chg_det_state,
                'speaker_state': speaker_state,
                'model_cfg': cfg
            }
            save_path = os.path.join(snapshot_dir,
                                     snapshot_file_format % t)
            save_checkpoint(checkpoint, save_path)

            print('Running eval at iter %d' % t)
            set_mode('eval', [change_detector, speaker])
            with torch.no_grad():
                test_iter_start_time = time.time()

                idx_to_word = train_dataset.get_idx_to_word()

                sent_save_dir = sent_dir
                if not os.path.exists(sent_save_dir):
                    os.makedirs(sent_save_dir)


                predictions = {}

                with open(os.path.join(output_dir, cfg.train.setting + '_result.txt'), 'w') as f:
                    f.write('epoch: ' + str(epoch) + '\n')
                for val_i, val_batch in enumerate(tqdm(val_loader)):
                    d_feats, sc_feats, labels, sc_pos_labels,  masks, pair_index, d_adj_matrix, q_adj_matrix, d_sem_adj_matrix, q_sem_dj_matrix,d_bb,q_bb, question = val_batch

                    val_batch_size = d_feats.size(0)

                    d_feats, sc_feats = d_feats.to(device), sc_feats.to(device)
                    question = question.to(device)
                    sc_pos_labels= sc_pos_labels.to(device)
                    labels, masks = labels.to(device), masks.to(device)
                    d_adj_matrix, q_adj_matrix = d_adj_matrix.to(device), q_adj_matrix.to(device)
                    d_sem_adj_matrix, q_sem_dj_matrix = d_sem_adj_matrix.to(device), q_sem_dj_matrix.to(device)

                    if args.setting == 'mode2':
                        d_adj_matrix = process_matrix(d_adj_matrix, cfg, d_feats.shape[1], d_feats.device, type='spatial')
                        q_adj_matrix = process_matrix(q_adj_matrix, cfg, sc_feats.shape[1], sc_feats.device, type='spatial')

                        d_sem_adj_matrix = process_matrix(d_sem_adj_matrix, cfg, d_feats.shape[1], d_feats.device,type='semantic')
                        q_sem_dj_matrix = process_matrix(q_sem_dj_matrix, cfg, sc_feats.shape[1], sc_feats.device,type='semantic')

                    chg_pos_logits, chg_pos_att_bef, chg_pos_att_aft, \
                    chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff = change_detector(d_feats, sc_feats, d_adj_matrix, q_adj_matrix,d_sem_adj_matrix, q_sem_dj_matrix,d_bb,q_bb, question, setting = cfg.train.setting, graph = args.graph)

                    speaker_output_pos, _ = speaker._sample(chg_pos_feat_bef,
                                                            chg_pos_feat_aft,
                                                            chg_pos_feat_diff,
                                                            labels, cfg, sample_max=1)

                    gen_sents_pos = decode_sequence(idx_to_word, speaker_output_pos)

                    chg_pos_att_bef = chg_pos_att_bef.cpu().numpy()
                    chg_pos_att_aft = chg_pos_att_aft.cpu().numpy()

                    dummy = np.ones_like(chg_pos_att_bef)
                    for val_j in range(speaker_output_pos.size(0)):
                        gts = decode_sequence(idx_to_word, labels[val_j][:,1:])
                        predict_sent = gen_sents_pos[val_j]

                        predictions[str(int(pair_index[val_j]))] = predict_sent



                        message = '%s results:\n' % str(pair_index[val_j])
                        message += '----------<Prediction>----------\n'
                        message += predict_sent + '\n'
                        message += '----------<GROUND TRUTHS>----------\n'
                        for gt in gts:
                            message += gt + '\n'
                        message += '===================================\n'
                        # print(message)
                        with open(os.path.join(output_dir,cfg.train.setting+'_result.txt'), 'a') as f:
                            f.write(message)

                # test_iter_end_time = time.time() - test_iter_start_time
                result_save_path_pos = os.path.join(sent_save_dir, 'eval_results_%s.json'%t)
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
                    if args.use_wandb:
                        wandb.log({metric: score})

                print('evaluated')

                if coco_eval.eval['Bleu_1'] > best:
                    best = coco_eval.eval['Bleu_1']
                    save_path = os.path.join(snapshot_dir, 'checkpoint_best.pt')
                    save_checkpoint(checkpoint, save_path)
                    print('Best checkpoint saved')
            set_mode('train', [change_detector, speaker])

    epoch += 1
