# Feature extraction
This folder provides the code necessary for extracting the features needed for our method.
## Prerequisite data
Prior to extraction, the following files need to be prepared:

1. MIMIC-CXR-JPG converted into 1024x1024 PNG images. These must be saved to the mimic-cxr-png folder. Run mimic_jpg2png() in converter.py.

```angular2html
python converter.py -p <input_path_to_mimic_cxr_jpg> -o <output_path_to_mimic_cxr_png>
```
After running this, you will obtain two files: “mimic_shape_full.pkl” and “dicom2id.pkl”. The former file contains the shape of each image, while the latter file contains the mapping between dicom id and the feature index.
2. Faster-rcnn checkpoints. Make sure these are located in the checkpoints folder.
   - `checkpoints/model_final_for_anatomy_gold.pth`  ([Download link](https://drive.google.com/file/d/1DFm94EFPyYdC_sPVApBAR-g0UefhdUQG/view?usp=sharing). It is used for anatomical structure detection and can be obtained by running train_anatomy.py)
   - `checkpoints/model_final_for_vindr.pth`  ([Download link](https://drive.google.com/file/d/15PayxjSodrS4X5uhn7fX0iycVx9U8uR8/view?usp=sharing). It is used for disease detection and can be obtained by running train-vindr-online.py)
3. Dictionary files. Make sure these are in the dictionary folder.
   - `dictionary/category_ana.pkl` (An anatomical structure category set)
   - `dictionary/GT_counting_adj.pkl` (A co-occurrence matrix of findings in mimic-cxr-jpg)
4. (Optional) If you choose to run the code to generate GT_counting_adj.pkl in step 3 by yourself, then run
```angular2html
python dictionary/preparation.py -p <path_to_mimic_cxr_jpg>
```


## Extraction

### 1, Anatomical structure feature extraction

```angular2html
python ana_bbox_generator.py
```

### 2, Disease feature extraction
the disease feature are extracted using pre-trained disease detection model, on the anatomical structures extracted in the previous step. 
```angular2html
python bbox_generator_by_location.py
```


### 3, Feature Combination

```angular2html
python combine_dicts.py
```
