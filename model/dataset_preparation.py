import json
import os
import pandas as pd
import pickle
from tqdm import tqdm
import nltk
import numpy as np
import h5py
import argparse
import warnings

def transform_pos_tag(tag_list, d_pos, max_seq):
    out = []
    for item in tag_list:
        tag = item[1]
        id = d_pos[d_pos['tag'] == tag]['id'].values[0]
        out.append(id)
    for i in range(len(out),max_seq):
        out.append(0)
    return out

def get_label(caption_list, max_seq):
    output = np.zeros(max_seq)
    output[:len(caption_list)] = np.array(caption_list)
    return output

def save_h5(questions, answers, pos, label_start_idx, label_end_idx, feature_idx,max_seq = 60, times=0, length = 100):
    filename = os.path.join('data/VQA_mimic_dataset.h5')
    if times == 0:
        h5f = h5py.File(filename, 'w')
        questions_dataset = h5f.create_dataset("questions", (length, 20),
                                                    maxshape=(None, 20),
                                                    chunks=(100, 20),
                                                    dtype='int64')
        answers_dataset = h5f.create_dataset("answers", (length, max_seq),
                                                    maxshape=(None, max_seq),
                                                    chunks=(100, max_seq),
                                                    dtype='int64')
        pos_dataset = h5f.create_dataset("pos", (length, max_seq),
                                                    maxshape=(None, max_seq),
                                                    chunks=(100, max_seq),
                                                    dtype='int64')
        label_start_idx_dataset = h5f.create_dataset("label_start_idx", (length, 1),
                                                    maxshape=(None,1),
                                                    chunks=(100, 1),
                                                    dtype='int64')
        label_end_idx_dataset = h5f.create_dataset("label_end_idx", (length,1),
                                                    maxshape=(None, 1),
                                                    chunks=(100, 1),
                                                    dtype='int64')
        feature_idx_dataset = h5f.create_dataset("feature_idx", (length,2),
                                                    maxshape=(None, 2),
                                                    chunks=(100, 2),
                                                    dtype='int64')
    else:
        h5f = h5py.File(filename, 'a')
        questions_dataset = h5f['questions']
        answers_dataset = h5f['answers']
        pos_dataset = h5f['pos']
        label_start_idx_dataset = h5f['label_start_idx']
        label_end_idx_dataset = h5f['label_end_idx']
        feature_idx_dataset = h5f['feature_idx']

    if len(questions) != length:
        adding = len(questions)
    else:
        adding = length

    questions_dataset.resize([times * length + adding, 20])
    questions_dataset[times * length:times * length + adding] = questions

    answers_dataset.resize([times*length+adding, max_seq])
    answers_dataset[times*length:times*length+adding] = answers

    pos_dataset.resize([times*length+adding, max_seq])
    pos_dataset[times*length:times*length+adding] = pos

    label_start_idx_dataset.resize([times*length+adding, 1])
    label_start_idx_dataset[times*length:times*length+adding] = label_start_idx

    label_end_idx_dataset.resize([times*length+adding,1])
    label_end_idx_dataset[times*length:times*length+adding] = label_end_idx

    feature_idx_dataset.resize([times*length+adding,2])
    feature_idx_dataset[times*length:times*length+adding] = feature_idx


    h5f.close()
def save_coco_format():
    print('start saving coco format')
    path_splits = 'data/splits_mimic_VQA.json'
    with open(path_splits, 'r')as f:
        splits = json.load(f)

    path = 'data/mimic_pair_questions.csv'
    df = pd.read_csv(path)
    anno_list= []
    image_list = []
    for name in ['train','val','test']:
        split = splits[name]
        for index in split:
        # for i in range(len(df_caption['captionAB'])):
            anno_record = {}
            image_record = {}
            try:
                anno_record['id'] = str(index)
                anno_record['image_id'] = str(index) # important
                anno_record['category_id'] = 0
                anno_record['caption'] = df['answer'][index]
                anno_record['question'] = df['question'][index]

                image_record['id'] = str(index)

                anno_list.append(anno_record)
                image_list.append(image_record)
            except:
                break
        dict ={}
        dict['info'] = []
        dict['licenses'] = []
        dict['categories'] = []
        dict['images'] = image_list
        dict['annotations'] = anno_list



        json.dump(dict, open('data/mimic_gt_captions_%s.json'%name, 'w'))
        image_list = []
        anno_list = []
        print('saved')
def transform_questions2dataset(simple = False, custom_dataset = False):
    '''
    output: h5:{question, answer, pos, start_idx, end_idx, feature_idx}
        vocabulary
        splits
    '''
    max_seq = 90
    length = 100 # every 5000 step to save dataset
    block_size = 100

    question_path = './data/mimic_pair_questions.csv'
    d = pd.read_csv(question_path)
    pos_path = 'data/POS.csv'
    d_pos = pd.read_csv(pos_path)
    with open('data/dicom2id.pkl', 'rb') as f:
        dicom2id = pickle.load(f) # index in bbx features
    with open('data/study2dicom.pkl', 'rb') as f:
        study2dicom = pickle.load(f)

    if custom_dataset:
        vocab = {'<start>':1}
    else:
        with open('data/vocab_mimic_VQA.json', 'rb') as f:
            vocab = json.load(f)
    questions = []
    answers = []
    label_start_idx = []
    label_end_idx = []
    feature_idx = []
    pos = []
    times = 0
    total = 0
    for i in tqdm(range(len(d))):
        if simple:
            if d.iloc[i]['question_type'] != 'difference':
                continue
        # if i < 212400:
        #     continue
        raw_question = d.iloc[i]['question']
        raw_answer = d.iloc[i]['answer']
        question_list = nltk.word_tokenize(raw_question.lower())
        answer_list = nltk.word_tokenize(raw_answer.lower())
        answer_list = ['<start>']+answer_list
        answer_pos = transform_pos_tag(nltk.pos_tag(answer_list), d_pos,max_seq)
        pos.append(answer_pos[:max_seq])
        for word in question_list + answer_list:
            if word not in vocab:
                vocab[word] = len(vocab) + 1
                warnings.warn('Unknown word: %s. Please be aware that this may result in the checkpoint we provided not functioning properly' %word)
        question_list = [vocab[word] for word in question_list]
        answer_list = [vocab[word] for word in answer_list]
        question = get_label(question_list, 20)
        answer = get_label(answer_list[:max_seq], max_seq)
        questions.append(question)
        answers.append(answer)
        label_start_idx.append([i])
        label_end_idx.append([i+1])
        study_id = d.iloc[i]['study_id']
        ref_id = d.iloc[i]['ref_id']

        feature_idx.append([dicom2id[study2dicom[study_id]], dicom2id[study2dicom[ref_id]]])
        if i == len(d)-1 or len(questions) == length:
            questions = np.array(questions)
            answers = np.array(answers)
            pos = np.array(pos)
            label_start_idx = np.array(label_start_idx)
            label_end_idx = np.array(label_end_idx)
            save_h5(questions, answers, pos, label_start_idx, label_end_idx, feature_idx, max_seq=max_seq, times=times, length=block_size)
            questions, answers, pos, label_start_idx, label_end_idx, feature_idx = [], [], [], [], [], []
            times += 1
        total += 1


    splits = {}
    indexes = np.arange(total).tolist()
    splits['train'] = indexes[0:int(np.ceil(0.8 * total))]
    splits['val'] = indexes[int(np.ceil(0.8 * total)):int(np.ceil(0.9 * total))]
    splits['test'] = indexes[int(np.ceil(0.9 * total)):]

    path_vocab = 'data/vocab_mimic_VQA.json'
    path_splits = 'data/splits_mimic_VQA.json'
    with open(path_vocab, 'w') as f:
        json.dump(vocab, f,indent=4)
    with open(path_splits, 'w') as f:
        json.dump(splits, f,indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--save_coco_format", action='store_true',  help="save coco format data for evaluation")
    parser.add_argument("-t", "--transform_dataset", action='store_true', help="transform questions and answers into the format that can be used in the model")
    args = parser.parse_args()
    if not args.save_coco_format and not args.transform_dataset:
        print('please choose at least one mode')
        return
    if args.transform_dataset:
        transform_questions2dataset()
    if args.save_coco_format:
        save_coco_format()


if __name__ == '__main__':
    main()