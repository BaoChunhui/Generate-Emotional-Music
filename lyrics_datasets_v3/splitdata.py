import numpy as np
import re
import csv
import tqdm
import time

import pdb

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

dataset_50 = np.load("dataset_100_v3.npy")

melody = dataset_50[0]
lyrics = dataset_50[1]
labels = dataset_50[2]

with_labels_id = []
for i in range(labels.shape[0]):
    if labels[i] != 'unlabelled':
        with_labels_id.append(i)

dataset_50_group = dataset_50[:, with_labels_id]
num = dataset_50_group.shape[1]
lenth = num // 8
print(num, lenth)
new_dataset = []
for i in range(8):
    new_dataset.append(dataset_50_group[:, (i*lenth):((i*lenth)+lenth)])
dataset_50_group_8 = np.array(new_dataset)
np.save("dataset_100_v3_clf.npy", dataset_50_group_8)

