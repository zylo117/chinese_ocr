import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

from tools.io_.hdf5datasetwriter import HDF5DatasetWriter

IMAGE_PATH = '../datasets/plate/'
HDF5_SAVE_PATH = '../datasets/license_plate_ocr/'
batchsize = 128
maxlabellength = 10
imagesize = (280, 32)
char_list = [s.strip(os.linesep) for s in open('./char_std_5990.txt', 'r').readlines()]

pl = os.listdir(IMAGE_PATH)
ipl = []
for p in pl:
    if p.endswith('.jpg'):
        ipl.append(p)

train_data, val_data = train_test_split(ipl, test_size=0.1, random_state=42)
# test_data = val_data

train_labels_sparse = []
for label in train_data:
    train_labels_sparse.append([char_list.index(l) for l in label.split('_')[0]])

val_labels_sparse = []
for label in val_data:
    val_labels_sparse.append([char_list.index(l) for l in label.split('_')[0]])

train_labels = np.ones([len(train_labels_sparse),
                        maxlabellength]) * -1  # multiply a number that is out of the character index. In this case, >5990 would do.
val_labels = np.ones([len(val_labels_sparse), maxlabellength]) * -1
train_label_length = len(train_labels)
val_label_length = len(val_labels)
input_length = imagesize[0] // 8

# caffe_ocr中把0作为blank，但是tf的CTC the last class is reserved to the blank label.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/ctc/ctc_loss_calculator.h
for i, train_label in enumerate(train_labels_sparse):
    train_labels[i, :len(train_label)] = [int(l) - 1 for l in train_label]
for i, val_label in enumerate(val_labels_sparse):
    val_labels[i, :len(val_label)] = [int(l) - 1 for l in val_label]

train_hdf5 = HDF5DatasetWriter((train_label_length, 32, 280, 1),
                               HDF5_SAVE_PATH + '/train.hdf5', maxLabelLength=10, overwrite=False)
val_hdf5 = HDF5DatasetWriter((val_label_length, 32, 280, 1),
                             HDF5_SAVE_PATH + '/val.hdf5', maxLabelLength=10, overwrite=False)

len_set = [train_label_length, val_label_length]
data_set = [train_data, val_data]
label_set = [train_labels, val_labels]
h5_set = [train_hdf5, val_hdf5]
for i, hdf5 in enumerate(h5_set):
    for d, l in zip(data_set[i], label_set[i]):
        img = cv2.imread(IMAGE_PATH + '/' + d, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 32), interpolation=cv2.INTER_LANCZOS4)
        img = cv2.copyMakeBorder(img, 0, 0, 0, 180, cv2.BORDER_CONSTANT, value=0)
        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        img = img / 255 - 0.5
        img = np.expand_dims(img, axis=2)
        hdf5.add([img], [l], [input_length], [len_set[i]])
        print('adding {}'.format(d))

    hdf5.close()

shutil.copy(HDF5_SAVE_PATH + '/val.hdf5', HDF5_SAVE_PATH + '/test.hdf5')
