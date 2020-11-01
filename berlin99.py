import pickle
import numpy as np
import os
from keras.optimizers import Adam, RMSprop
from keras import regularizers


# def load_data():
#     f = open('./ASVP_speech2.pkl', 'rb')
#     train_feature, test_feature, train_labels, test_seg_labels = pickle.load(f)
#     return train_feature, test_feature, train_labels, test_seg_labels
#
# train_feature, test_feature, train_labels,test_seg_labels = load_data()
#
# # train_feature_mel=np.expand_dims(train_sample_feature,axis=-1)
# # test_feature_mel=np.expand_dims(test_sample_feature,axis=-1)
# print("train:",train_feature.shape,"test:",test_feature.shape)
# def load_data():
#     f = open('./sound_features1.pkl', 'rb')
#     train_feature, test_feature, train_labels, test_seg_labels,train_address,test_address=pickle.load(f)
#     return train_feature, test_feature, train_labels, test_seg_labels,train_address,test_address
def load_data():
    f = open('./features_speech_c2_folder-st-4.pkl', 'rb')
    train_feature, test_feature, train_labels, test_seg_labels=pickle.load(f)
    return train_feature, test_feature, train_labels, test_seg_labels

################获取特征，标签，地址
train_feature, test_feature, train_labels, test_seg_labels=load_data()
# train_feature, test_feature, train_labels, test_seg_labels,train_address,test_address=load_data()
print(train_feature.shape,":", test_feature.shape,":", train_labels.shape,":", test_seg_labels.shape)
# print(train_feature.shape,":", test_feature.shape,":", train_labels.shape,":", test_seg_labels.shape,":",train_address.shape,":",test_address.shape)
# train_features=np.expand_dims(train_feature,axis=1)
# print(train_features.shape)
# test_features=np.expand_dims(test_feature,axis=1)
# print(test_features.shape)

# train_feature=np.add.reduce(train_feature,3)
# test_feature=np.add.reduce(test_feature,3)#np.squeeze(train_feature_mel, axis=(3,)).shape
print(train_feature.shape,":",test_feature.shape)
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, Activation, GRU
from keras.layers import Conv2D, concatenate, MaxPooling2D, Flatten, Embedding, Lambda
import tensorflow as tf
from keras.layers.convolutional import Conv2D,MaxPooling2D#,GlobalAveragePooling2D
from keras.models import Model,Sequential
from keras import optimizers
from keras.layers import Input,Conv1D,BatchNormalization,MaxPooling1D,LSTM,Dense,Activation,Layer

from keras.utils import to_categorical
import keras.backend as K
import argparse
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import Dropout, Dense, TimeDistributed, BatchNormalization
import librosa
import sklearn
import os
import numpy as np
import re
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten,Conv2D,MaxPooling2D,GlobalAveragePooling2D,AveragePooling2D
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import load_model
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, KFold
from keras.optimizers import Adam, RMSprop,SGD
from keras.models import model_from_json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # python 的方式指定GPU id

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True  # 按需设置显存利用
#tf.reset_default_graph()

import numpy as np
import os
from os.path import isfile
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, Activation, GRU
from keras.layers import Conv2D, concatenate, MaxPooling2D, Flatten, Embedding, Lambda

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop

from keras import regularizers

import librosa
import librosa.display
import matplotlib.pyplot as plt


class Configure(object):
    def __init__(self):
        self.win_len = 0.025  # 帧长25ms
        self.win_step = 0.01  # 帧移10ms
        self.feature_dim = 40  # 特征维度

        self.num_fc = 64
        self.batch_size = 32
        self.num_epochs = 1500  # best model will be saved before number of epochs reach this value
        self.learning_rate = 0.0001
        self.decay = 1e-6
        self.momentum = 0.9


config = Configure()
#####################
# batch_size = 32
num_classes = 6
n_features = train_feature.shape[2]
n_time = train_feature.shape[1]
####################
nb_filters1 = 64
nb_filters2 = 64
nb_filters3 = 128
nb_filters4 = 128
nb_filters5 = 128
ksize = (3, 3)
pool_size_1 = (2, 2)
pool_size_2 = (2, 2)
pool_size_3 = (4, 4)

dropout_prob = 0.10
dense_size1 = 128
lstm_count = 96
num_units = 56

BATCH_SIZE = 50
EPOCH_COUNT = 200
L2_regularization = 0.001


def conv_recurrent_model_build(model_input):
    print('Building model...')
    layer = model_input

    ### Convolutional blocks
    conv_1 = Conv2D(filters=nb_filters1, kernel_size=ksize, strides=1,
                    kernel_regularizer=regularizers.l2(L2_regularization),
                    padding='same', name='conv_1')(layer)
    bn_1 = BatchNormalization()(conv_1)
    act_1 = Activation("elu")(bn_1)
    print(conv_1.shape)
    pool_1 = MaxPooling2D(pool_size_1, strides=(2, 2))(act_1)
    print("1", pool_1.shape)

    conv_2 = Conv2D(filters=nb_filters2, kernel_size=ksize, strides=1,
                    kernel_regularizer=regularizers.l2(L2_regularization),
                    padding='same', name='conv_2')(pool_1)
    bn_2 = BatchNormalization()(conv_2)
    act_2 = Activation("elu")(bn_2)
    print(conv_2.shape)
    pool_2 = MaxPooling2D(pool_size_1, strides=(2, 2))(act_2)
    print("2", pool_2.shape)

    conv_3 = Conv2D(filters=nb_filters4, kernel_size=ksize, strides=1,
                    kernel_regularizer=regularizers.l2(L2_regularization),
                    padding='same', name='conv_3')(pool_2)
    bn_3 = BatchNormalization()(conv_3)
    act_3 = Activation("elu")(bn_3)
    print(conv_3.shape)
    pool_3 = MaxPooling2D(pool_size_1, strides=(2, 2))(act_3)
    print("3", pool_3.shape)

    conv_4 = Conv2D(filters=nb_filters4, kernel_size=ksize, strides=1,
                    kernel_regularizer=regularizers.l2(L2_regularization),
                    padding='same', name='conv_4')(pool_3)
    bn_4 = BatchNormalization()(conv_4)
    act_4 = Activation("elu")(bn_4)

    print(conv_4.shape)
    pool_4 = MaxPooling2D(pool_size_3, strides=(4, 4))(act_4)
    print("4", pool_4.shape)

    #     conv_5 = Conv2D(filters = nb_filters5, kernel_size = ksize, strides=1,
    #                       padding= 'valid', activation='relu', name='conv_5')(pool_4)
    #     print(conv_5.shape)
    #     pool_5 = MaxPooling2D(pool_size_2)(conv_5)
    #     print("5",pool_5.shape)

    flatten1 = Flatten()(pool_4)
    ### Recurrent Block

    # Pooling layer
    pool_lstm1 = MaxPooling2D(pool_size_3, strides=(4, 4), name='pool_lstm')(layer)

    # Embedding layer

    squeezed = Lambda(lambda x: K.squeeze(x, axis=-1))(pool_lstm1)
    #     flatten2 = K.squeeze(pool_lstm1, axis = -1)
    #     dense1 = Dense(dense_size1)(flatten)

    # Bidirectional GRU
    lstm = LSTM(256)(squeezed)  # default merge mode is concat

    # Concat Output
    concat = concatenate([flatten1, lstm], axis=-1, name='concat')

    ## Softmax Output
    output = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(L2_regularization),
                   name='preds')(concat)

    model_output = output
    model = Model(model_input, model_output)

    opt = optimizers.SGD(lr=config.learning_rate, decay=config.decay, momentum=config.momentum, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #     opt = Adam(lr=0.001)
    # opt = RMSprop(lr=0.0005)  # Optimizer
    # model.compile(
    #     loss='categorical_crossentropy',
    #     optimizer=opt,
    #     metrics=['accuracy']
    # )

    print(model.summary())
    return model


def train_model(x_train, y_train, x_val, y_val):
    n_frequency = x_train.shape[2]
    n_frames = x_train.shape[1]
    # reshape and expand dims for conv2d

    input_shape = (n_frames, n_frequency, 1)
    model_input = Input(input_shape, name='input')

    model = conv_recurrent_model_build(model_input)
    n_fold=4
    early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=100, mode="auto")
    save_best_model = keras.callbacks.ModelCheckpoint(
        "./model/best_model_speech_st-con2d.hdf5" + str(n_fold) + '_' + 'weights.{epoch:02d}-{accuracy:.4f}-{val_accuracy:.4f}.hdf5',
        monitor='val_accuracy', verbose=0, save_best_only=True,
        save_weights_only=False, mode='auto', period=1)

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)
    # mc = ModelCheckpoint('./New/best_model_con2d.hdf5', monitor='val_categorical_accuracy',
    #                      mode='max', verbose=1, save_best_only=True)
    #
    model_json = model.to_json()
    with open("./model/model_speech_st-con2d.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('./model/model_speech_st-con2d_weight.h5')
    print("save model to disk")
    model.save("./model/model_speech_st-con2d.h5")
    # Fit the model and get training history.
    print('Training...')
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
                        validation_data=(x_val, y_val), verbose=1, callbacks=[early_stop,save_best_model])

    return model, history


model, history  = train_model(train_feature,train_labels,test_feature,test_seg_labels)

import matplotlib.pyplot as plt
def show_summary_stats(history):
    # List all data in history
    print(history.history.keys())

    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_summary_stats(history)
from sklearn.metrics import classification_report
# config = Config()
# dict_genres=config.emotion_list

emotion_dict = {
        "neutral": "03",
        "happy": "03",
        "sad": "04",
        # "angry": "05",
        "fearful": "06",
        # "disgust": "07",
        "surprised": "08"
    }

# emotion_dict = {
#         "happy": "03",
#         "sad": "04",
#         # "angry": "05",
#         "fearful": "06",
#         # "disgust": "07",
#         "surprised": "08",
#         "speech_sound": "00"
#     }

y_true = np.argmax(test_seg_labels, axis = 1)
X_valid = np.expand_dims(test_feature, axis = -1)
y_pred = model.predict(test_feature)
y_pred = np.argmax(y_pred, axis=1)
labels = [2,3,4,6,8]
target_names = emotion_dict.keys()

print(y_true.shape, y_pred.shape)
print(classification_report(y_true, y_pred, target_names=target_names))

from sklearn.metrics import confusion_matrix
import seaborn as sns


mat = confusion_matrix(y_true, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=emotion_dict.keys(),
            yticklabels=emotion_dict.keys())
plt.xlabel('true label')
plt.ylabel('predicted label')

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_true, y_pred))

