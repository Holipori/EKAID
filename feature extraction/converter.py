import pydicom
import os
import cv2
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import pickle
import argparse

import PIL # optional

# originally, this file is for convert dicom files to png files.
# but then the convert_mimic() function is for getting shape before the conversion.


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)

    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)

    return im


def read_xray(path, voi_lut=True, fix_monochrome=True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data

def convert_folder():
    # Specify the .dcm folder path
    folder_path = "/home/xinyue/dataset/vinbigdata/test"
    # Specify the output jpg/png folder path
    jpg_folder_path = "/home/xinyue/dataset/vinbigdata-png/test"
    # images_path = os.listdir(folder_path)
    for n, image in enumerate(images_path):

        xray = read_xray(os.path.join(folder_path, image))
        im = resize(xray, size=1024)
        PNG = True

        if PNG == False:
            image = image.replace('.dicom', '.jpg')
        else:
            image = image.replace('.dicom', '.png')
        # cv2.imwrite(os.path.join(jpg_folder_path, image), image)
        im.save(os.path.join(jpg_folder_path, image))
        if n % 50 == 0:
            print('{} image converted'.format(n))

def mimic_jpg2png(data_path, out_path):
    data_path = os.path.join(data_path, '2.0.0/files')
    p_folder = os.listdir(data_path)
    size = 1024
    n = 0
    dict = []
    mimic_shapeid = {}
    # with open("/home/xinyue/VQA_ReGat/dicom_list.pkl", "rb") as tf:
    #     dicom_list = pickle.load(tf)
    for p_fold in p_folder:
        if p_fold == 'index.html':
            continue
        p_path = os.path.join(data_path, p_fold)
        pp_folder = os.listdir(p_path)
        for pp_fold in pp_folder:
            if pp_fold == 'index.html':
                continue
            pp_path = os.path.join(p_path,pp_fold)
            if not os.path.isdir(pp_path):
                continue
            s_folder = os.listdir(pp_path)
            for s_fold in s_folder:
                if s_fold == 'index.html':
                    continue
                s_path = os.path.join(pp_path, s_fold)
                if not os.path.isdir(s_path):
                    continue
                files = os.listdir(s_path)
                for file in files:
                    if file == 'index.html':
                        continue
                    new_filename = os.path.join(out_path, file.replace('.jpg', '.png'))
                    record = {}
                    file_path = os.path.join(s_path,file)
                    im = Image.open(file_path)
                    record['image'] = file.replace('.jpg','')
                    record['height'] = im.size[0]
                    record['width'] = im.size[1]
                    dict.append(record)
                    mimic_shapeid[record['image']] = n
                    if os.path.exists(new_filename):
                        pass
                    else:
                        im = im.resize((size, size), Image.LANCZOS)
                        im.save(new_filename)
                    n += 1
                    if n % 50 == 0:
                        print('{} image converted'.format(n))
    if not os.path.exists('data'):
        os.mkdir('data')
    with open('data/mimic_shape_full.pkl', 'wb') as f:
        pickle.dump(dict,f)
        print('file saved')
    with open('data/mimic_shapeid_full.pkl', 'wb') as f:
        pickle.dump(mimic_shapeid,f)
        print('file saved')



# xray = read_xray("/home/xinyue/dataset/vinbigdata/test/c07a2a2fae1462d1b0dc1bd308adf2d8.dicom")
# im = resize(xray, size=1024)
# im.save("/home/xinyue/dataset/vinbigdata-png/test/c07a2a2fae1462d1b0dc1bd308adf2d8.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--mimic_path", type=str, default=None, required=True, help="path to mimic-cxr-jpg dataset")
    parser.add_argument("-o", "--out_path", type=str, default=None, required=True, help="path to output png dataset")
    args = parser.parse_args()
    mimic_jpg2png(data_path = args.mimic_path, out_path = args.out_path)

if __name__ == '__main__':
    main()