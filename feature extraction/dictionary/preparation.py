import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
import argparse

def get_from_GT_label(datafolder):
    path = os.path.join(datafolder,'2.0.0/files/mimic-cxr-2.0.0-chexpert.csv.gz')
    df = pd.read_csv(path)

    diseases_list = df.columns[2:16].values

    counting_adj = np.zeros([14, 14], int)
    for idx in tqdm(range(len(df))):
        for i in range(len(diseases_list)):
            for j in range(i,len(diseases_list)):
                dis1 = df.iloc[idx][diseases_list[i]]
                dis2 = df.iloc[idx][diseases_list[j]]
                if dis1 == 1 and dis2 == 1:
                    counting_adj[i,j] += 1
                    counting_adj[j,i] += 1

    counting_adj = counting_adj / np.linalg.norm(counting_adj)
    counting_adj = np.array(counting_adj)
    with open('dictionary/GT_counting_adj.pkl', 'wb') as f:
        pickle.dump(counting_adj, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--data_path", type=str, default=None, required=True, help="path to mimic-cxr-jpg dataset")
    args = parser.parse_args()
    get_from_GT_label(args.data_path)

if __name__ == '__main__':
    main()