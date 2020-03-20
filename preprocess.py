# """
# Load the 'nii' file and save as pkl file.
# Carefully check your path please.
# """

# import os
# import pickle

# import nibabel as nib
# import numpy as np

# from utils import Parser

# args = Parser()
# modalities = ('flair', 't1ce', 't1', 't2')

# train_set = {
#     'root': './data/2018/MICCAI_BraTS_2018_Data_Training',
#     'flist': 'all.txt',
# }

# valid_set = {
#     'root': './data/2018/MICCAI_BraTS_2018_Data_Validation',
#     'flist': 'valid.txt',
# }

# test_set = {
#     'root': './data/2018/MICCAI_BraTS_2018_Data_TTest',
#     'flist': 'test.txt',
# }


# def nib_load(file_name):
#     if not os.path.exists(file_name):
#         return np.array([1])

#     proxy = nib.load(file_name)
#     data = proxy.get_data()
#     proxy.uncache()
#     return data


# def normalize(image, mask=None):
#     assert len(image.shape) == 3  # shape is [H,W,D]
#     assert image[0, 0, 0] == 0  # check the background is zero
#     if mask is not None:
#         mask = (image > 0)  # The bg is zero

#     mean = image[mask].mean()
#     std = image[mask].std()
#     image = image.astype(dtype=np.float32)
#     image[mask] = (image[mask] - mean) / std
#     return image


# def savepkl(data, path):
#     with open(path, 'wb') as f:
#         pickle.dump(data, f)


# def process_f32(path):
#     """ Set all Voxels that are outside of the brain mask to 0"""
#     label = np.array(nib_load(path + 'seg.nii.gz'), dtype='uint8', order='C')
#     images = np.stack([
#         np.array(nib_load(path + modal + '.nii.gz'), dtype='float32', order='C')
#         for modal in modalities], -1)

#     mask = images.sum(-1) > 0

#     for k in range(4):
#         x = images[..., k]  #
#         y = x[mask]  #

#         lower = np.percentile(y, 0.2)  # 算分位数
#         upper = np.percentile(y, 99.8)

#         x[mask & (x < lower)] = lower
#         x[mask & (x > upper)] = upper

#         y = x[mask]

#         x -= y.mean()
#         x /= y.std()

#         images[..., k] = x

#     output = path + 'data_f32.pkl'
#     print("saving:", output)
#     savepkl(data=(images, label), path=output)
#     return images, label


# def doit(dset):
#     root = dset['root']
#     file_list = os.path.join(root, dset['flist'])
#     subjects = open(file_list).read().splitlines()
#     names = [sub.split('/')[-1] for sub in subjects]
#     paths = [os.path.join(root, sub, name + '_') for sub, name in zip(subjects, names)]
#     for path in paths:
#         process_f32(path)


# doit(train_set)
# doit(valid_set)
# # doit(test_set)




"""
The code will split the training set into k-fold for cross-validation
"""

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
import shutil

root = './data/2018/MICCAI_BraTS_2018_Data_Training'
valid_data_dir = './data/2018/MICCAI_BraTS_2018_Data_Validation'

backup = './2018/datasets'
backup_files = os.listdir(backup)
if len(backup_files) != 0:
    print("Copy from backup")
    for file in backup_files:
        shutil.copy(os.path.join(backup, file), os.path.join(root, file))
        count=0
        with open(os.path.join(root, file), 'r') as f:
            for line in f:
                count += 1
            print("File {} has {} lines.".format(file, count))
    sys.exit()

def write(data, fname, root=root):
    fname = os.path.join(root, fname)
    with open(fname, 'w') as f:
        f.write('\n'.join(data))

limit = float(sys.argv[1])

hgg = os.listdir(os.path.join(root, 'HGG'))
hgg = [os.path.join('HGG', f) for f in hgg]
lgg = os.listdir(os.path.join(root, 'LGG'))
lgg = [os.path.join('LGG', f) for f in lgg]

print("Original size: HGG:{}, LGG:{}, Total:{}".format(len(hgg), len(lgg), len(hgg) + len(lgg)))
hgg = hgg[:int(limit*len(hgg))]
lgg = lgg[:int(limit*len(lgg))]
print("Limited size: HGG:{}, LGG:{}, Total:{}".format(len(hgg), len(lgg), len(hgg) + len(lgg)))
X = hgg + lgg
Y = [1] * len(hgg) + [0] * len(lgg)

write(X, 'all.txt')
shutil.copy(os.path.join(root,'all.txt'), os.path.join(backup, 'all.txt'))
X, Y = np.array(X), np.array(Y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)

for k, (train_index, valid_index) in enumerate(skf.split(Y, Y)):
    train_list = list(X[train_index])
    valid_list = list(X[valid_index])

    write(train_list, 'train_{}.txt'.format(k))
    write(valid_list, 'valid_{}.txt'.format(k))

    shutil.copy(os.path.join(root,'train_{}.txt'.format(k)),
                            os.path.join(backup, 'train_{}.txt'.format(k)))
    shutil.copy(os.path.join(root,'valid_{}.txt'.format(k)), 
                            os.path.join(backup, 'valid_{}.txt'.format(k)))

valid = os.listdir(os.path.join(valid_data_dir))
valid = [f for f in valid if not (f.endswith('.csv') or f.endswith('.txt'))]
write(valid, 'valid.txt', root=valid_data_dir)