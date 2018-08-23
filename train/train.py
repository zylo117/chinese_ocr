# -*- coding:utf-8 -*-

import os

from network import densenet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import json
import threading
import numpy as np
from PIL import Image

import tensorflow as tf
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.utils.training_utils import multi_gpu_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten
from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, Adamax
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from importlib import reload
import network

img_h = 32
img_w = 280
batch_size = 1
maxlabellength = 7


def get_session(gpu_fraction=1.0):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """

    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batchsize):
        r_n = []
        if (self.index + batchsize > self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index: self.index + batchsize]
            self.index = self.index + batchsize

        return r_n


def gen(data_file, image_path, batchsize=128, maxlabellength=10, imagesize=(32, 280)):
    image_label = readfile(data_file)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        # shufimagefile = _imagefile[r_n.get(batchsize // 4)]
        # shufimagefile = np.repeat(shufimagefile, 4)
        for i, j in zip(range(batchsize), shufimagefile):
            img = cv2.imread(os.path.join(image_path, j), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 32), interpolation=cv2.INTER_LANCZOS4)
            img = cv2.copyMakeBorder(img, 0, 0, 0, 180, cv2.BORDER_CONSTANT, value=0)
            # if i % 4 == 0:
            #     img = cv2.resize(img, (100, 32), interpolation=cv2.INTER_LANCZOS4)
            #     img = cv2.copyMakeBorder(img, 0, 0, 0, 180, cv2.BORDER_CONSTANT, value=0)
            # elif i % 4 == 1:
            #     img = cv2.resize(img, (86, 32), interpolation=cv2.INTER_LANCZOS4)
            #     img = cv2.copyMakeBorder(img, 0, 0, 0, 194, cv2.BORDER_CONSTANT, value=0)
            # elif i % 4 == 2:
            #     img = cv2.resize(img, (114, 32), interpolation=cv2.INTER_LANCZOS4)
            #     img = cv2.copyMakeBorder(img, 0, 0, 0, 166, cv2.BORDER_CONSTANT, value=0)
            # elif i % 4 == 3:
            #     img = cv2.resize(img, (75, 24), interpolation=cv2.INTER_LANCZOS4)
            #     img = cv2.copyMakeBorder(img, 4, 4, 0, 205, cv2.BORDER_CONSTANT, value=0)

            img = img / 255 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)
            str = image_label[j]
            label_length[i] = len(str)

            if len(str) <= 0:
                print("len < 0", j)
            input_length[i] = imagesize[1] // 8
            labels[i, :len(str)] = [int(k) - 1 for k in str]

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  }
        outputs = {'ctc': np.zeros([batchsize])}
        yield (inputs, outputs)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(img_h, nclass):
    input = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet.dense_cnn(input, nclass)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input, labels, input_length, label_length], outputs=loss_out)
    # model = multi_gpu_model(model, gpus=2)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adamax(), metrics=['accuracy'])

    return basemodel, model


if __name__ == '__main__':
    modelPath = './models/weights-densenet-97-1.96.h5'
    train_txt = '../datasets/license_plate_ocr/new_train.txt'
    val_txt = '../datasets/license_plate_ocr/new_val.txt'
    image_path = '../datasets/plate'
    EPOCH = 100
    try:
        INITIAL_EPOCH = int(modelPath.split('-')[-2]) + 1
    except:
        INITIAL_EPOCH = 0
    TRAIN_IMAGE_COUNT = len(open(train_txt).readlines())
    VAL_IMAGE_COUNT = len(open(val_txt).readlines())


    # lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch  # 1-10, batchsize 128
    # lr_schedule = lambda epoch: 0.0005 * 0.8 ** epoch  # 11-20
    # lr_schedule = lambda epoch: 0.0005 * 0.95 ** epoch  # 20-30
    # lr_schedule = lambda epoch: 0.001 * 0.95 ** epoch  # 30-37
    # lr_schedule = lambda epoch: 0.0005 * 0.95 ** epoch  # 38-44
    # lr_schedule = lambda epoch: 0.0005 * 0.90 ** epoch  # 44-48
    # lr_schedule = lambda epoch: 0.0001 * 0.8 ** epoch  # 49-54
    # lr_schedule = lambda epoch: 0.00001 * 1 ** epoch  # 54-60
    # lr_schedule = lambda epoch: 0.000001 * 1 ** epoch  # 54-60
    # lr_schedule = lambda epoch: 0.001 * 1 ** epoch  # 60-67, batchsize 64
    # lr_schedule = lambda epoch: 0.0001 * 1 ** epoch  # 67-74
    # lr_schedule = lambda epoch: 0.00005 * 1 ** epoch  # 75-79
    # lr_schedule = lambda epoch: 0.00001 * 1 ** epoch  # 80-92
    # lr_schedule = lambda epoch: 0.000001 * 1 ** epoch  # 92-98, batchsize 8
    lr_schedule = lambda epoch: 0.0001 * 1 ** epoch  # 98-100, batchsize 1


    char_set = open('char_std_5990.txt', 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
    nclass = len(char_set)

    K.set_session(get_session())
    reload(network)
    basemodel, model = get_model(img_h, nclass)

    if os.path.exists(modelPath):
        print("Loading model weights...")
        basemodel.load_weights(modelPath)
        print('done!')

    train_loader = gen(train_txt, image_path, batchsize=batch_size,
                       maxlabellength=maxlabellength,
                       imagesize=(img_h, img_w))
    test_loader = gen(val_txt, image_path, batchsize=batch_size,
                      maxlabellength=maxlabellength,
                      imagesize=(img_h, img_w))

    checkpoint = ModelCheckpoint(filepath='./models/weights-densenet-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss',
                                 save_best_only=False, save_weights_only=True)

    learning_rate = np.array([lr_schedule(i) for i in range(EPOCH)])
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)

    print('-----------Start training-----------')
    model.fit_generator(train_loader,
                        steps_per_epoch=TRAIN_IMAGE_COUNT // batch_size,
                        epochs=EPOCH,
                        initial_epoch=97,
                        validation_data=test_loader,
                        validation_steps=VAL_IMAGE_COUNT // batch_size,
                        callbacks=[checkpoint, earlystop, changelr, tensorboard])
