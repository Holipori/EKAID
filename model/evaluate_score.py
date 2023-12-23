import json

from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap
from evaluation import my_COCOEvalCap
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse

annotation_file = 'data/mimic_gt_captions_test.json'


def finding_best_timing(results_file):
    if '.json' in results_file:
        dir = results_file[:results_file.find('eval_results')]
    else:
        raise ValueError('results_file should be a json file')
    files = os.listdir(dir)
    best_b1 = 0
    for f in tqdm(files[10:]):
        file_path = os.path.join(dir, f)
        with open(file_path, 'r') as f:
            results = json.load(f)
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        coco = COCO(annotation_file)
        coco_result = coco.loadRes(file_path)
        coco_eval = my_COCOEvalCap(coco, coco_result)
        coco_eval.params['image_id'] = coco_result.getImgIds()
        coco_eval.evaluate()

        bleu1 = list(coco_eval.eval.items())[0][1]
        if bleu1 > best_b1:
            best_b1 = bleu1
            best_file = file_path

    print(best_file, best_b1)


def caption_metric(input_file = None):
    if input_file is None:
        input_file = results_file

    with open(input_file, 'r') as f:
        results = json.load(f)

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    coco = COCO(annotation_file)
    coco_result = coco.loadRes(input_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = my_COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    output = []
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
        output.append(score)
    return output

def find_best_result():
    with open(results_file, 'r') as f:
        results = json.load(f)

    with open('data/mimic_gt_captions_test.json', 'r') as f:
        annotations = json.load(f)

    imageid2id = {}

    for i in range(len(annotations['annotations'])):
        imageid2id[annotations['annotations'][i]['image_id']] = i

    for i in range(len(results)):
        caption = results[i]['caption'].replace(' .', '.'). replace(' ,', ',').replace(' ?', '?').replace(' !', '!').replace(' ;', ';').replace(' :', ':').replace('  ', ' ')
        imageid = results[i]['image_id']
        gt_caption = annotations['annotations'][imageid2id[imageid]]['caption']
        if caption in gt_caption:
            print('yes')
        if 'opacity' not in gt_caption:
            if 'edema' in caption and 'edema' in gt_caption:
                print(i, caption, imageid)
            if 'pneumonia' in caption and 'pneumonia' in gt_caption:
                print(i, caption, imageid)
            if 'consolidation' in caption and 'consolidation' in gt_caption:
                print(i, caption, imageid)
            if 'atelectasis' in caption and 'atelectasis' in gt_caption:
                print(i, caption, imageid)



def caption_metric_by_question_type(input_file = None, target_type = 'location'):
    path = 'data/mimic_pair_questions.csv'
    df = pd.read_csv(path)


    if input_file is None:
        input_file = results_file

    with open(input_file, 'r') as f:
        results = json.load(f)

    new_results = []
    for i in tqdm(range(len(results))):
        image_id = results[i]['image_id']
        question_type = df.iloc[int(image_id)]['question_type']
        if question_type == target_type:
            new_results.append(results[i])

    with open('experiments/temp.json', 'w') as f:
        json.dump(new_results, f)

    input_file = 'experiments/temp.json'

    coco = COCO(annotation_file)
    coco_result = coco.loadRes(input_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = my_COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    output = []
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
        output.append(score)
    return output

def caption_metric_by_abnormality(input_file = None):
    path = 'data/mimic_pair_questions.csv'
    df = pd.read_csv(path)


    if input_file is None:
        input_file = results_file

    with open(input_file, 'r') as f:
        results = json.load(f)

    disease_path = 'data/question_gen/lib/disease_lib.csv'
    disease_df = pd.read_csv(disease_path)
    disease_list = disease_df['official_name'].tolist()

    disease2id = { disease_list[i]: i for i in range(len(disease_list))}

    preds = []
    gts = []
    # for disease in disease_list:
    for i in tqdm(range(len(results))):
        image_id = results[i]['image_id']
        answer = results[i]['caption']
        question = df.iloc[int(image_id)]['question']
        gt_answer = df.iloc[int(image_id)]['answer']
        if question == 'what abnormalities are seen in this image?':
            preds_onehot = np.zeros(len(disease_list))
            gts_onehot = np.zeros(len(disease_list))
            gt_diss = gt_answer.split(',')
            pred_ans = answer.split(',')
            for dis in gt_diss:
                if dis in disease2id:
                    gts_onehot[disease2id[dis]] = 1
            for dis in pred_ans:
                if dis in disease2id:
                    preds_onehot[disease2id[dis]] = 1
            preds.append(preds_onehot)
            gts.append(gts_onehot)

    preds = np.array(preds)
    gts = np.array(gts)

    for i in range(len(disease_list)):
        num_i = np.sum(gts[:, i])
        correct_i = 0
        for j in range(len(gts)):
            if gts[j, i] == 1 and preds[j, i] == 1:
                correct_i += 1


        acc = correct_i / num_i

        print(disease_list[i], acc)

    new_preds = []
    new_gts = []
    # remove the columns with no gt
    for i in range(gts.shape[1]):
        if np.sum(gts[:, i]) > 0:
            new_preds.append(preds[:, i])
            new_gts.append(gts[:, i])
    new_preds = np.array(new_preds)
    new_gts = np.array(new_gts)

    auc = roc_auc_score(new_gts.transpose(), new_preds.transpose(), average=None)
    print('auc', auc)





def acc(results_file):
    gt_path = annotation_file
    with open(gt_path, 'r') as f:
        gt = json.load(f)
    pr_path = results_file
    with open(pr_path, 'r') as f:
        pr = json.load(f)
    gt = gt['annotations']

    total_correct = 0
    open_correct = 0
    closed_correct = 0
    total = 0
    opend = 0
    closed = 0
    for i in range(len(gt)):
        if 'what has changed' in gt[i]['question']:
            continue
        gt_ans = gt[i]['caption']
        pr_ans = pr[i]['caption']
        total += 1
        if gt_ans == 'yes' or gt_ans == 'no':
            closed += 1
        else:
            opend += 1
        if gt_ans == pr_ans:
            total_correct += 1
            if gt_ans == 'yes' or gt_ans == 'no':
                closed_correct += 1
            else:
                open_correct += 1
    print('total', total_correct/total)
    print('open', open_correct/opend)
    print('closed', closed_correct/closed)

    return total_correct/total, open_correct/opend , closed_correct/closed


def get_multiple():
    file_path_pre = 'experiments/temp/mode2_location_0.0001_2022-05-19-00-18-37/eval_sents/eval_results_%s.json'
    for i in range(80000,100000,10000):
        print(i)

def find_the_best(runname):
    best = 0
    best_i = 0
    for i in range(2000,60000, 2000):
        try:
            file = 'experiments/temp/%s/eval_sents/eval_results_%s.json'%(runname, i)
            total_acc, open_acc, closed_acc = acc(file)
            if total_acc > best:
                best = total_acc
                best_i = i
        except:
            pass
    print('final', best_i, best)

def combine_json():
    test_file = 'data/mimic_gt_captions_test.json'
    val_file = 'data/mimic_gt_captions_val.json'
    train_file = 'data/mimic_gt_captions_train.json'
    with open(test_file, 'r') as f:
        test = json.load(f)
    with open(val_file, 'r') as f:
        val = json.load(f)
    with open(train_file, 'r') as f:
        train = json.load(f)
    test['annotations'] = test['annotations'] + val['annotations'] + train['annotations']
    with open('data/question_gen/mimic_gt_captions.json', 'w') as f:
        json.dump(test, f)

def move_folders():
    folder_list = ['mode2_location_semantic_0.0001_coef0.333000-0.333000_2022-11-12-20-57-32',
                'mode2_location_spatial_0.0001_coef0.333000-0.333000_2022-11-13-00-15-25',
                'mode2_location_implicit_0.0001_coef0.333000-0.333000_2022-11-13-03-36-21',
                'mode2_location_all_0.0001_coef0.333000-0.333000_2022-11-13-08-04-56']

    for folder in folder_list:
        os.system('mv experiments/temp/%s experiments/final/'%folder)

def caption_metric_by_question_type(input_file = None, target_type = 'location'):
    path = 'data/mimic_pair_questions.csv'
    df = pd.read_csv(path)


    if input_file is None:
        raise ValueError('input_file is None')

    with open(input_file, 'r') as f:
        results = json.load(f)

    new_results = []
    for i in tqdm(range(len(results))):
        image_id = results[i]['image_id']
        question_type = df.iloc[int(image_id)]['question_type']
        if question_type == target_type:
            new_results.append(results[i])

    with open('experiments/temp.json', 'w') as f:
        json.dump(new_results, f)

    input_file = 'experiments/temp.json'

    coco = COCO(annotation_file)
    coco_result = coco.loadRes(input_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = my_COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    coco_eval.evaluate()

    # print output evaluation scores
    output = []
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
        output.append(score)
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--run_name", type=str, default=None, required=True,
                        help="name of the running")
    parser.add_argument("-c", "--checkpoint_num", type=int, default=None, required=True, help="checkpoint number")
    parser.add_argument('-t', '--target_type', type=str, default='', help='target type', examples=['location', 'abnormality', 'difference'])
    args = parser.parse_args()
    if args.target_type:
        print('Evaluating %s'%args.target_type)
        results = caption_metric_by_question_type(input_file='experiments/temp/final/%s/eval_sents/eval_results_%s.json' % (args.run_name, str(args.checkpoint_num)), target_type='difference')
    else:
        print('Evaluating all')
        results = caption_metric(input_file='experiments/temp/final/%s/eval_sents/eval_results_%s.json'%(args.run_name, str(args.checkpoint_num)))


if __name__ == '__main__':
    main()
