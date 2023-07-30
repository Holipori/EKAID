import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from utils.mimic_utils import purify:
import h5py
import sys
import cv2
from detectron2.utils.visualizer import Visualizer
# sys.path.append("..")
sys.path.append("../../faster_rcnn")
from train_vindr import my_predictor

# this file is for the old difference captioning

path_caption = os.path.join('..', 'data', 'mimic_diff_caption.csv')
df = pd.read_csv(path_caption)
path = '/home/xinyue/dataset/mimic/mimic_all.csv'
df_all = pd.read_csv(path)
with open('/home/xinyue/dataset/mimic/study2dicom.pkl', 'rb') as f:
    study2dicom = pickle.load(f)

finding_name = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
                'Pneumothorax', 'Support Devices']

draw_bbx = True
id = 2
# id = np.random.randint(0,785396)
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
    predictor, vindr_metadata = my_predictor(0.3)
    im = cv2.imread(path1)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],metadata=vindr_metadata,scale=1.0,)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img1 = out.get_image()[:, :, ::-1]

    im = cv2.imread(path2)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],metadata=vindr_metadata,scale=1.0,)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
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

