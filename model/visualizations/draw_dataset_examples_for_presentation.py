import pickle
import json
import pandas as pd
import os
import random
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.mimic_utils import purify

def find_report(study_id):
    path = '/home/xinyue/dataset/mimic/mimic_all.csv'
    df_all = pd.read_csv(path)
    subject_id = df_all[df_all['study_id'] == int(study_id)]['subject_id'].values[0]
    report_path = '/home/xinyue/dataset/mimic_reports'
    p1 = 'p' + str(subject_id)[:2]
    p2 = 'p'+str(subject_id)
    report_name = 's' + str(int(study_id)) + '.txt'
    with open(os.path.join(report_path, p1, p2, report_name), 'r') as f:
        report = f.read()

    report = report.replace('\n', '').replace('FINDINGS', '\nFINDINGS').replace('IMPRESSION', '\nIMPRESSION')
    print(report)
    return report


def purify(array):
    for i in range(len(array)):
        if np.isnan(array[i]):
            array[i] = 0
    return array

def get_mimic_ori_label(study_id):
    finding_name = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                    'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                    'Pneumothorax', 'Support Devices']
    finding = df_all[df_all['study_id'] == int(study_id)].iloc[0][2:16].values

    findings =  {'def_findings': [], 'possible_findings': [] , 'def_no_findings': []}
    for i in range(len(finding)):
        if finding[i] == 1:
            findings['def_findings'].append(finding_name[i])
        elif finding[i] == 0:
            findings['def_no_findings'].append(finding_name[i])
        elif finding[i] == -1:
            findings['possible_findings'].append(finding_name[i])
    return findings

def radom_sample_diff_pair_from_csv():
    dataset_path = '../data/mimic_pair_questions_small.csv'
    dataset = pd.read_csv(dataset_path)
    sample_list_idx = random.sample(range(len(dataset)), 1)

    img_dir = '/home/xinyue/dataset/mimic-cxr-png'
    with open('/home/xinyue/dataset/mimic/study2dicom.pkl', 'rb') as f:
        study2dicom = pickle.load(f)
    disease_path = '../data/all_diseases.json'
    with open(disease_path, 'r') as f:
        diseases = json.load(f)
    studyid2idx = {}
    for i, d in enumerate(diseases):
        studyid2idx[d['study_id']] = i

    for i in tqdm(range(len(sample_list_idx))):
        print()
        print('================================')
        print()
        print('Sample', i)
        # random select a main-reference pair
        record = dataset.iloc[sample_list_idx[i]]
        study_id = str(int(float(record['study_id'])))
        subject_id = str(int(float(record['subject_id'])))
        ref_id = str(int(float(record['ref_id'])))



        main_report = find_report(study_id)
        ref_report = find_report(ref_id)

        main_dis = list(diseases[studyid2idx[study_id]]['entity'])
        main_no_dis = diseases[studyid2idx[study_id]]['no_entity']
        ref_dis = list(diseases[studyid2idx[ref_id]]['entity'])
        ref_no_dis = diseases[studyid2idx[ref_id]]['no_entity']

        main_ori_label = get_mimic_ori_label(study_id)
        ref_ori_label = get_mimic_ori_label(ref_id)

        print('-----------------')
        print('main report: ' + main_report.replace('\n', ' '))
        print('-----------------')
        print('Ours')
        print('main disease: ' + str(main_dis))
        print('main no disease: ' + str(main_no_dis))
        print('-----------------')
        print('MIMIC-CXR')
        print('main disease: ' + str(main_ori_label['def_findings']))
        print('main no disease: ' + str(main_ori_label['def_no_findings']))
        print('main possible disease: ' + str(main_ori_label['possible_findings']))
        print('-----------------')
        print('ref report: ' + ref_report.replace('\n', ' '))
        print('-----------------')
        print('Ours')
        print('ref disease: ' + str(ref_dis))
        print('ref no disease: ' + str(ref_no_dis))
        print('-----------------')
        print('MIMIC-CXR')
        print('ref disease: ' + str(ref_ori_label['def_findings']))
        print('ref no disease: ' + str(ref_ori_label['def_no_findings']))
        print('ref possible disease: ' + str(ref_ori_label['possible_findings']))

        # find all pairs of this main-reference pair
        records = dataset[dataset['study_id'] == int(study_id)][dataset['ref_id'] == int(ref_id)]
        for j in range(len(records)):
            question = records.iloc[j]['question']
            answer = records.iloc[j]['answer']
            print('-----------------')
            print('Question%d: %s'% (j,records.iloc[j]['question']))
            print('Answer: ' + records.iloc[j]['answer'])

        # loading image
        name1 = study2dicom[int(study_id)] + '.png'
        path1 = os.path.join(img_dir, name1)
        img1 = cv2.imread(path1)
        name2 = study2dicom[int(ref_id)] + '.png'
        path2 = os.path.join(img_dir, name2)
        img2 = cv2.imread(path2)

        # add text
        output_text = 'Question: ' + question
        output_text += '\n\n'
        output_text += 'GT Answer: ' + answer

        # plot image
        fig = plt.figure(figsize=(10, 8), dpi=300)
        plt.axis('off')
        plt.title('index: ' + str(i))
        plt.text(0, -0.05, output_text, fontsize=12, wrap=True)

        ax1 = fig.add_subplot(1, 2, 2)
        ax1.imshow(img1)
        ax1.set_title('Main ' + str(study_id))
        ax2 = fig.add_subplot(1, 2, 1)
        ax2.imshow(img2)
        ax2.set_title('Reference ' + str(ref_id))
        plt.show()

def check_any_in(list, text):
    for word in list:
        if word in text:
            return word
    return False

def check_target(entities, report, target):
    if target == 'probability':
        if len(entities) == 0:
            return False
        n = 0
        for key in entities:
            ent = entities[key]
            if ent['probability_score'] != -3 and ent['probability_score'] != 3:
                n += 1
        if n/len(entities) > 0.5:
            return True
    elif target == 'location':
        if len(entities) == 0:
            return False
        n = 0
        for key in entities:
            ent = entities[key]
            if ent['location'] is not None or ent['post_location'] is not None:
                n += 1
        if n/len(entities) > 0.5:
            return True
    elif target == 'inference':
        keyword_list = ['indicat', 'suggest', 'reflect', 'represent', 'explain'] #todo: use lib
        if check_any_in(keyword_list, report):
            return True
    elif target == 'exclude':
        keyword_list = ['exclude', 'rule out', 'ruled out'] #todo: use lib
        if check_any_in(keyword_list, report):
            return True
    return False

def find_examples(study_id = None, target = 'location'):
    img_dir = '/home/xinyue/dataset/mimic-cxr-png'
    keyinfo_path = '../data/all_diseases.json'
    with open(keyinfo_path, 'r') as f:
        keyinfo = json.load(f)
    # random shuffle  keyinfo
    random.shuffle(keyinfo)

    if study_id is not None:
        for record in tqdm(keyinfo):
            if record['study_id'] == str(study_id):
                keyinfo = [record]
                break
    for record in keyinfo:
        study_id = record['study_id']
        entities = record['entity']
        report = find_report(study_id)
        if check_target(entities, report, target):
            print(study_id)
            print('Report: \n', report.replace('\n', ' '))
            print('\nFindings: ')
            for key in entities:
                ent = entities[key]
                location = ' '.join(ent['location']) if ent['location'] is not None else ent['post_location']
                print("<%s> \"%s\", %s, %s"%(key, ent['probability'], ent['probability_score'], location))

            # plot image
            name = record['dicom_id'] + '.png'
            path = os.path.join(img_dir, name)
            img = cv2.imread(path)
            plt.title('Study ID: ' + str(study_id))
            plt.imshow(img)
            plt.show()
            return

    print('a')

def check_if_target_sample(diseases, target_key='inference'):
    for disease in diseases:
        if diseases[disease][target_key] != []:
            return True
    return False

def radom_sample_from_keyinfo():
    dataset_path = '../data/all_diseases.json'
    dataset = pd.read_json(dataset_path)
    sample_list_idx = random.sample(range(len(dataset)), 10)

    img_dir = '/home/xinyue/dataset/mimic-cxr-png'
    # studyid2idx = {}
    # for i, d in enumerate(diseases):
    #     studyid2idx[d['study_id']] = i

    for i in tqdm(range(len(sample_list_idx))):
        diseases = dataset.iloc[sample_list_idx[i]]['entity']
        if not check_if_target_sample(diseases, 'infer'):
            continue

        print()
        print('================================')
        print()
        print('Sample', i)
        # random select a main-reference pair
        record = dataset.iloc[sample_list_idx[i]]
        study_id = str(int(float(record['study_id'])))
        subject_id = str(int(float(record['subject_id'])))
        report = find_report(study_id)

        print('-----------------')
        print('main report: ' + report.replace('\n', ' '))
        print('-----------------')
        print('Ours')
        for disease in diseases:
            print(disease)
            for key in diseases[disease]:
                if diseases[disease][key] is not None:
                    print('- ', key, ':', diseases[disease][key])
        print('-----------------')
        print('MIMIC-CXR')
        main_ori_label = get_mimic_ori_label(study_id)
        print('main disease: ' + str(main_ori_label['def_findings']))
        print('main no disease: ' + str(main_ori_label['def_no_findings']))
        print('main possible disease: ' + str(main_ori_label['possible_findings']))


        # loading image
        name1 = record['dicom_id'] + '.png'
        path1 = os.path.join(img_dir, name1)
        img1 = cv2.imread(path1)


        # plot image
        fig = plt.figure(dpi=300)
        plt.axis('off')
        plt.title('index: ' + str(sample_list_idx[i]) + '\n' + 'Study ID: ' + str(study_id))
        # plt.text(0, -0.05, output_text, fontsize=12, wrap=True)


        plt.imshow(img1)
        plt.show()


if __name__ == '__main__':
    find_report(51105845)
    # path = '/home/xinyue/dataset/mimic/mimic_all.csv'
    # df_all = pd.read_csv(path)
    #
    # # radom_sample_diff_pair_from_csv()
    # radom_sample_from_keyinfo()


    # find_examples(study_id= 59551271, target='exclude')