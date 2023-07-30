import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

from tqdm import tqdm
import h5py
import sys
import cv2
import json
from detectron2.utils.visualizer import Visualizer
# sys.path.append('..')
sys.path.append('/home/xinyue/faster_rcnn')
from train_vindr import my_predictor
# from train_anatomy import my_predictor
import spacy
import re

def purify(array):
    for i in range(len(array)):
        if np.isnan(array[i]) or array[i] == -1:
            array[i] = 0
    return array
def main():
    path_caption = os.path.join('..', '../data', 'mimic_diff_caption.csv')
    df = pd.read_csv(path_caption)
    path = '/home/xinyue/dataset/mimic/mimic_all.csv'
    df_all = pd.read_csv(path)
    with open('/home/xinyue/dataset/mimic/study2dicom.pkl', 'rb') as f:
        study2dicom = pickle.load(f)

    finding_name = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                    'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                    'Pneumothorax', 'Support Devices']

    draw_bbx = False
    id = 632977
    # id = np.random.randint(0,785396)
    # for i in range(785396):
    #     image1 = int(df.iloc[id]['imageA'])
    #     image2 = int(df.iloc[id]['imageB'])
    #     if image1 == 50284501 and image2 == 55956986:
    #         id = i
    #         break

    print(study2dicom[int(df.iloc[id]['imageA'])])


    path = '/home/xinyue/faster-rcnn/output/vindr_box/ana_bbox_features.hdf5'
    hf = h5py.File(path, 'r')
    # hf[''][self.feature_idx[img_idx, 0]]


    # loading image
    name1 = study2dicom[int(df.iloc[id]['imageA'])] + '.png'
    name2 = study2dicom[int(df.iloc[id]['imageB'])] + '.png'
    path1 = os.path.join('/home/xinyue/dataset/mimic-cxr-png',name1)
    path2 = os.path.join('/home/xinyue/dataset/mimic-cxr-png',name2)
    img1 = mpimg.imread(path1)
    img2 = mpimg.imread(path2)
    img1 = img1 * -1 + 1
    img2 = img2 * -1 + 1

    hf.close()

    # draw disease bbx
    if draw_bbx:
        predictor, metadata = my_predictor(0.3)
        im = cv2.imread(path1)
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],metadata=metadata,scale=1.0,)
        out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        img1 = out.get_image()[:, :, ::-1]

        im = cv2.imread(path2)
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],metadata=metadata,scale=1.0,)
        out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        img2 = out.get_image()[:, :, ::-1]
        # plt.figure(dpi=300)
        # plt.imshow(out.get_image()[:, :, ::-1])
        # string = 'Prediction'
        # plt.title(string)
        # plt.show()



    # find subject id, study id, caption
    subject_id = df.iloc[id]['subject_id']
    image1 = int(df.iloc[id]['imageA'])
    image2 = int(df.iloc[id]['imageB'])
    caption = df.iloc[id]['captionAB']
    n = 80
    if len(caption) > n:
        caption = caption.split('.')
        caption = caption[0] + '.\n' + caption[1]
    view = df.iloc[id]['view']
    if view == 'postero-anterior':
        view = 'PA view'
    elif view == 'antero-posterior':
        view = 'AP view'

    # get labels of each
    finding1 = purify(df_all[df_all['study_id'] == image1].iloc[0][2:16].values)
    finding2 = purify(df_all[df_all['study_id'] == image2].iloc[0][2:16].values)
    findings1 = ''
    findings2 = ''
    for i in np.nonzero(finding1)[0]:
        findings1+= finding_name[i] +'\n'
    for i in np.nonzero(finding2)[0]:
        findings2+= (finding_name[i]) + '\n'


    # final plotting
    fig = plt.figure(figsize=(10, 8), dpi=300)
    plt.axis('off')
    plt.text(0.5,0.88,'Subject id: '+ str(int(subject_id)),horizontalalignment='center', fontsize = 'x-large')
    plt.text(0.9,0.88,view,horizontalalignment='center', fontsize = 'x-large', color= 'red')
    plt.text(0,0,'caption: '+ caption,horizontalalignment='left', fontsize = 'large')
    ax1 = fig.add_subplot(121)
    ax1.axis('off')
    ax1.title.set_text('Image A: '+ str(image1))
    ax1.text(0,1030,'label: '+ findings1, verticalalignment='top')
    ax1.imshow(img1,cmap='Greys')
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    ax2.title.set_text('Image B: '+ str(image2))
    ax2.text(0,1030,'label: '+ findings2, verticalalignment='top')
    ax2.imshow(img2,cmap='Greys')
    plt.savefig('../plotted/'+str(subject_id)+'_'+str(image1)+'_'+str(image2)+'.png', bbox_inches='tight')
    plt.show()

    print('Subject id:', subject_id)
    print('image A:', image1)
    print('image B:', image2)
    print('label A:', findings1)
    print('label B:', findings2)
    print('caption:', caption)


    report_path = '/home/xinyue/dataset/mimic_reports'
    p1 = 'p'+str(subject_id)[:2]
    p2 = 'p'+str(int(subject_id))
    report1_name = 's'+ str(int(image1))+'.txt'
    report2_name = 's'+ str(int(image2))+'.txt'

    with open(os.path.join(report_path,p1,p2,report1_name),'r') as f:
        report1 = f.read()
    with open(os.path.join(report_path,p1,p2,report2_name),'r') as f:
        report2 = f.read()

    print('report A: \n'+ report1)
    print('\n')
    print('report B: \n'+ report2)

    nlp = spacy.load('en_ner_bc5cdr_md')
    doc = nlp(report1.replace('\n',' ').replace('   ', ' ').replace('  ',' '))
    print('ents:', doc.ents)

def main2():
    path_caption = os.path.join('..', '../data', 'question_gen', 'mimic_pair_questions_backup.csv')
    df = pd.read_csv(path_caption)
    # path = '/home/xinyue/dataset/mimic/mimic_all.csv'
    # df_all = pd.read_csv(path)
    with open('/home/xinyue/dataset/mimic/study2dicom.pkl', 'rb') as f:
        study2dicom = pickle.load(f)

    path_diseases = '../data/all_diseases.json'
    with open(path_diseases, 'r') as f:
        diseases_json = json.load(f)
    diseases_df = pd.DataFrame(diseases_json)


    # finding_name = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
    #                 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
    #                 'Pneumothorax', 'Support Devices']

    draw_bbx = False

    old_id = 0

    for id in tqdm(range(len(df))):
        question = df.iloc[id]['question']
        if 'what has changed in the' not in question:
            continue
        # study_id = df.iloc[id]['study_id']
        # if study_id == old_id:
        #     old_id = study_id
        #     continue
        # old_id = study_id
        print(study2dicom[int(df.iloc[id]['study_id'])])


        # path = '/home/xinyue/faster-rcnn/output/vindr_box/ana_bbox_features.hdf5'
        # hf = h5py.File(path, 'r')
        # hf[''][self.feature_idx[img_idx, 0]]


        # loading image
        name1 = study2dicom[int(df.iloc[id]['study_id'])] + '.png'
        name2 = study2dicom[int(df.iloc[id]['ref_id'])] + '.png'
        path1 = os.path.join('/home/xinyue/dataset/mimic-cxr-png',name1)
        path2 = os.path.join('/home/xinyue/dataset/mimic-cxr-png',name2)
        img1 = mpimg.imread(path1)
        img2 = mpimg.imread(path2)
        img1 = img1 * -1 + 1
        img2 = img2 * -1 + 1

        if len(list(diseases_df[diseases_df['study_id'] == str(df.iloc[id]['study_id'])]['entity'].values[0])) != 1 or len(list(diseases_df[diseases_df['study_id'] == str(df.iloc[id]['ref_id'])]['entity'].values[0])) != 1:
            continue

        # hf.close()

        # draw disease bbx
        if draw_bbx:
            try:
                predictor, metadata = my_predictor()
            except:
                pass
            im = cv2.imread(path1)
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],metadata=metadata,scale=1.0,)
            out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            img1 = out.get_image()[:, :, ::-1]

            im = cv2.imread(path2)
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],metadata=metadata,scale=1.0,)
            out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            img2 = out.get_image()[:, :, ::-1]
            # plt.figure(dpi=300)
            # plt.imshow(out.get_image()[:, :, ::-1])
            # string = 'Prediction'
            # plt.title(string)
            # plt.show()



        # find subject id, study id, caption
        subject_id = df.iloc[id]['subject_id']
        image1 = int(df.iloc[id]['study_id'])
        image2 = int(df.iloc[id]['ref_id'])

        question = df.iloc[id]['question']
        answer = df.iloc[id]['answer']
        n = 80
        if len(answer) > n:
            answer = answer.split('.')
            answer = answer[0] + '.\n' + answer[1]

        # get labels of each
        finding1 = list(diseases_df[diseases_df['study_id'] == str(image1)]['entity'].values[0])
        finding2 = list(diseases_df[diseases_df['study_id'] == str(image1)]['entity'].values[0])
        findings1 = '\n'.join(finding1)
        findings2 = '\n'.join(finding2)

        # final plotting
        fig = plt.figure(figsize=(10, 8), dpi=300)
        plt.axis('off')
        plt.text(0.5,0.88,'Subject id: '+ str(int(subject_id)),horizontalalignment='center', fontsize = 'x-large')
        # plt.text(0.9,0.88,view,horizontalalignment='center', fontsize = 'x-large', color= 'red')
        plt.text(0,0.1,'question: '+ question,horizontalalignment='left', fontsize = 'large')
        plt.text(0,0,'answer: '+ answer,horizontalalignment='left', fontsize = 'large')
        ax1 = fig.add_subplot(121)
        ax1.axis('off')
        ax1.title.set_text('Image A: '+ str(image1))
        ax1.text(0,1030,'label: '+ findings1, verticalalignment='top')
        ax1.imshow(img1,cmap='Greys')
        ax2 = fig.add_subplot(122)
        ax2.axis('off')
        ax2.title.set_text('Image B: '+ str(image2))
        ax2.text(0,1030,'label: '+ findings2, verticalalignment='top')
        ax2.imshow(img2,cmap='Greys')
        # plt.savefig('../plotted/'+str(subject_id)+'_'+str(image1)+'_'+str(image2)+'.png', bbox_inches='tight')
        plt.show()

        print('Subject id:', subject_id)
        print('image A:', image1)
        print('image B:', image2)
        # print('caption:', caption)


        report_path = '/home/xinyue/dataset/mimic_reports'
        p1 = 'p'+str(subject_id)[:2]
        p2 = 'p'+str(int(subject_id))
        report1_name = 's'+ str(int(image1))+'.txt'
        report2_name = 's'+ str(int(image2))+'.txt'

        with open(os.path.join(report_path,p1,p2,report1_name),'r') as f:
            report1 = f.read()
        with open(os.path.join(report_path,p1,p2,report2_name),'r') as f:
            report2 = f.read()

        print('report A: \n'+ report1)
        print('\n')
        print('report B: \n'+ report2)

        nlp = spacy.load('en_ner_bc5cdr_md')
        doc = nlp(report1.replace('\n',' ').replace('   ', ' ').replace('  ',' '))
        print('ents:', doc.ents)

def plot_by_id(study_id):
    report = find_report(study_id)
    print('Report:', study_id)
    print(report)

    # our entities
    disease_path = '/home/xinyue/SRDRL/data/datasets/all_diseases.json'
    with open(disease_path, 'r') as f:
        diseases = json.load(f)
    entities = []
    for record in diseases:
        if record['study_id'] == str(study_id):
            ents = record['entity']
            if ents != []:
                for ent in ents:
                    entities.append(ent)
    print('Our extracted entities: ', ', '.join(entities))
    print('\n')

    # our questions
    qa_path = '/home/xinyue/SRDRL/data/datasets/mimic_pair_questions.csv'
    df_qa = pd.read_csv(qa_path)
    d = df_qa[df_qa['study_id'] == study_id]
    print('Our generated questions:')
    for i in range(len(d)):
        print(d.iloc[i]['question'])
        print('GT Answer:', d.iloc[i]['answer'])

    # plotting part
    path = '/home/xinyue/dataset/mimic/mimic_all.csv'
    df_all = pd.read_csv(path)
    with open('/home/xinyue/dataset/mimic/study2dicom.pkl', 'rb') as f:
        study2dicom = pickle.load(f)
    finding_name = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                    'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                    'Pneumothorax', 'Support Devices']


    # loading image
    name1 = study2dicom[int(study_id)] + '.png'
    # name2 = study2dicom[int(df.iloc[id]['imageB'])] + '.png'
    path1 = os.path.join('/home/xinyue/dataset/mimic-cxr-png',name1)
    # path2 = os.path.join('/home/xinyue/dataset/mimic-cxr-png',name2)
    img1 = mpimg.imread(path1)
    # img2 = mpimg.imread(path2)
    img1 = img1 * -1 + 1



    image1 = int(study_id)

    # get labels of each
    finding1 = purify(df_all[df_all['study_id'] == image1].iloc[0][2:16].values)
    findings1 = ''
    for i in np.nonzero(finding1)[0]:
        findings1+= finding_name[i] +'\n'
    print('Mimic labels: ', ', '.join(findings1.split('\n')))


    # final plotting
    fig = plt.figure(figsize=(10, 8), dpi=300)
    plt.axis('off')
    ax1 = fig.add_subplot(111)
    ax1.axis('off')
    ax1.title.set_text('Image A: '+ str(image1))
    ax1.text(0,1030,'label: '+ findings1, verticalalignment='top')
    ax1.imshow(img1,cmap='Greys')
    plt.show()




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

    report.replace('\n', '').replace('FINDINGS', '\nFINDINGS').replace('IMPRESSION', '\nIMPRESSION')
    return report

def find_by_question():
    path = '../data/mimic_pair_questions.csv'
    d = pd.read_csv(path)
    type = 'abnormality'
    records = d[d['question_type'] == type]
    record = dict(records.iloc(np.random.randint(len(records))))

def go_over_all_reports():
    report_path = '/home/xinyue/dataset/mimic_reports'
    target = 0
    total = 0
    for p1 in tqdm(os.listdir(report_path)):
        if p1[0] != 'p':
            continue
        for p2 in tqdm(os.listdir(os.path.join(report_path,p1))):
            if p2[0] != 'p':
                continue
            for report_name in os.listdir(os.path.join(report_path,p1,p2)):
                if report_name.endswith('.txt'):
                    with open(os.path.join(report_path,p1,p2,report_name),'r') as f:
                        report = f.read()
                    # if 'HISTORY' in report:
                    #     # find strings from 'HISTORY' to'\n' using regex
                    #     history = re.findall(r'(HISTORY.*?)\n', report, re.DOTALL)
                    #
                    #     print(history)
                    #     target += 1

                    sentences = report.split('.')
                    for sent in sentences:
                        # if "edema" in sent and 'effusion' in sent and "cause" in sent:
                        #     print(report)
                        #     target += 1
                        if 'cardiomegaly' in sent  and 'edema' in sent and ("exaggerat" in sent or 'increase' in sent or 'cause' in sent or 'indicat' in sent):
                            print(report)
                            target += 1
                    total += 1
        print('target/total:', target / total)
    print('target:', target)
    print('total:', total)
    print('target/total:', target/total)

if __name__=='__main__':
    # main()
    # find_by_question()

    path = '/home/xinyue/dataset/mimic/mimic_all.csv'
    df_all = pd.read_csv(path)
    # random sample one from df_all
    index = np.random.randint(len(df_all))
    study_id = df_all.iloc[index]['study_id']

    id = 50852973
    plot_by_id(id)
    # plot_by_id(50159745)
    report = find_report(study_id=id)
    print(report)

    # go_over_all_reports()



