import pandas as pd
import numpy as np
import json
import cv2
import os
import pickle
import re
import math

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score

import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetLarge
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator



def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def get_preproc_function(scale_range = (0, 1), histogram_equalization = True):
    
    if histogram_equalization:
        def preproc(x):
            he_x = hisEqulColor(x)
            scale_x = ((he_x / 255.0) - float(scale_range[0]))*float(scale_range[1])
            return scale_x
    else:
        def preproc(x):
            scale_x = ((x / 255.0) - float(scale_range[0]))*float(scale_range[1])
            return scale_x
        
    return preproc

# 取出部分比例的資料id
def take_test_id(data, random_state , target_col, test_size = 0.2, ):
    d = shuffle(data)
    test_len = len(data)*test_size
    n_unique = data[target_col].nunique()
    test_index = d.groupby(target_col).head(int(test_len/n_unique)).index
    return test_index

# 自定義的generator function 目的在於批次讀入資料&前處理，避免記憶體爆掉
def image_data_generator(df, img_dir, file_col, target_col, batch_size, re_size, preprocess_function = None, augmentation = False, shuffle_ = True):
    
    while True:
        if shuffle_:
            data = shuffle(df)
        else:
            data = df

        x_batch = []; y_batch = []

        if augmentation:
            datagen=ImageDataGenerator(rotation_range=20,
                                       width_shift_range=0.1, 
                                       height_shift_range=0.1, 
                                       brightness_range=None, 
                                       shear_range=0.0, 
                                       zoom_range=[0.8, 1.2], 
                                       channel_shift_range=0.0, 
                                       cval=0.0, 
                                       rescale=None)

        for i in range(len(data)):
            img = cv2.imread(os.path.join(img_dir, data[file_col].iloc[i]))
            target = pd.get_dummies(data[target_col])
            img = cv2.resize(img, tuple(re_size))

            if preprocess_function:
                img = preprocess_function(img)
            if augmentation:
                img = datagen.random_transform(img)

            x_batch.append(img)
            y_batch.append(target.iloc[i].values)

            if (len(x_batch)==batch_size)|(i==(len(data)-1)):
                out_x = np.array(x_batch)
                out_y = np.array(y_batch)

                x_batch = [];y_batch = []
                yield out_x, out_y


# 定義 test_generator的函數，與先前不同的是在此我們是提供一個路徑，路徑中的所有圖片檔案皆會輸入模型做預測
def generator_from_dir(image_path, target_size, batch_size, preprocess_function = None):
    
    image_list = [i for i in os.listdir(image_path) if re.search('(png|jpg)', i)]
    img_list = []; file_list = []
    
    print('{} images found in directory'.format(len(image_list)))
    
    if image_path[-1]!='/':
        image_path +='/'
    
    for i in image_list:
        img = cv2.imread(image_path + i)
        img = cv2.resize(img, target_size)
        img = preprocess_function(img)
        
        img_list.append(img)
        file_list.append(i)
        
        if len(img_list)==batch_size:
            out_batch = img_list
            out_file = file_list
            
            img_list = []; file_list = []
            yield out_batch, out_file
    yield img_list, file_list
    


# load data
def get_and_save_data(params_dict):
    if type(params_dict['data_params']['df_dir']) is str:
        
        data = pd.read_csv(params_dict['data_params']['df_dir'])
        
        # 取出2成資料id作為測試資料id
        test_id = take_test_id(data, random_state = params_dict['train_params']['random_state'], target_col = params_dict['data_params']['target_col'])
        
        train_dat = data.drop(test_id)
        test_dat = data.loc[test_id]
        
        if params_dict['train_params']['early_stop_round']!=None:
            # 打亂並再度切割資料作為訓練與驗證集 (8 : 2)
            t_dat, v_dat = train_test_split(train_dat, shuffle = True, test_size = 0.2, random_state = params_dict['train_params']['random_state'])

            print(t_dat.shape)
            print(v_dat.shape)
        else:
            t_dat = train_dat
            v_dat = []
            print(train_dat.shape)
        
    elif type(params_dict['data_params']['df_dir']) is dict:
        
        t_dat = params_dict['data_params']['df_dir']['training_set']
        v_dat = params_dict['data_params']['df_dir']['validation_set']
        test_dat = params_dict['data_params']['df_dir']['testing_set']

    dat_dict = {'training_set': t_dat,
                'validation_set': v_dat,
                'testing_set': test_dat}
    
    if params_dict['data_params']['data_save_path']!='':
        pickle.dump(dat_dict,open(params_dict['data_params']['data_save_path'], 'wb'))
    
    return dat_dict


def build_model(params_dict, n_class):
    if '/' in params_dict['model_params']['pretrain_model']:
        model = load_model(params_dict['model_params']['pretrain_model'])
    else:
        imagenet_pretrain_model = params_dict['model_params']['pretrain_model'].split(':')[1]

        if imagenet_pretrain_model=='ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=params_dict['train_params']['img_shape'])
        elif imagenet_pretrain_model=='VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=params_dict['train_params']['img_shape'])
        elif imagenet_pretrain_model=='InceptionV3':
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=params_dict['train_params']['img_shape'])
        elif imagenet_pretrain_model=='InceptionResNetV2':
            base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=params_dict['train_params']['img_shape'])
        elif imagenet_pretrain_model=='DenseNet201':
            base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=params_dict['train_params']['img_shape'])
        elif imagenet_pretrain_model=='NASNetLarge':
            base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=params_dict['train_params']['img_shape'])
        else:
            print('parameter in config.model_params.pretrain_model should be either a path or imagenet:\'pretain_model_name\'')

        # layer freeze:
        if params_dict['train_params']['freeze_layer']:
            for i in params_dict['train_params']['freeze_layer'] :
                base_model.layers[i].trainable = False

        x = base_model.output
        x = Flatten()(x)
        x = Dropout(params_dict['train_params']['dropout_rate'])(x)

        if params_dict['train_params']['weight_regularization']:
            reg = keras.regularizers.l1_l2(l1 = params_dict['train_params']['weight_regularization']['l1'],
                                                   l2 = params_dict['train_params']['weight_regularization']['l2'])
        else:
            reg = None
        predictions = Dense(n_class, activation='softmax', kernel_regularizer=reg)(x)


        model = Model(inputs=base_model.input, outputs=predictions)

    if params_dict['train_params']['optimizer']['optimizer']=='SGD':
        optimizer = keras.optimizers.SGD(lr=params_dict['train_params']['optimizer']['lr'],
                                         momentum=params_dict['train_params']['optimizer']['momentum'],
                                         decay=params_dict['train_params']['optimizer']['decay'],
                                         nesterov=params_dict['train_params']['optimizer']['nesterov'])
    elif params_dict['train_params']['optimizer']['optimizer']=='RMSprop':
        optimizer = keras.optimizers.RMSprop(lr=params_dict['train_params']['optimizer']['lr'],
                                             rho=params_dict['train_params']['optimizer']['rho'],
                                             epsilon=params_dict['train_params']['optimizer']['epsilon'],
                                             decay=params_dict['train_params']['optimizer']['decay'])
    elif params_dict['train_params']['optimizer']['optimizer']=='Adagrad':
        optimizer = keras.optimizers.Adagrad(lr=params_dict['train_params']['optimizer']['lr'],
                                             epsilon=params_dict['train_params']['optimizer']['epsilon'],
                                             decay=params_dict['train_params']['optimizer']['decay'])
    elif params_dict['train_params']['optimizer']['optimizer']=='Adam':
        optimizer = keras.optimizers.Adam(lr=params_dict['train_params']['optimizer']['lr'],
                                          epsilon=params_dict['train_params']['optimizer']['epsilon'],
                                          decay=params_dict['train_params']['optimizer']['decay'])

    model.compile(loss=params_dict['train_params']['loss'],
                  optimizer=optimizer, metrics=['accuracy'])
    
    model_path = params_dict['model_params']['model_dir']+'/{}.h5'.format(params_dict['model_params']['model_name'])

    if params_dict['train_params']['early_stop_round']!=None:
        earlystop = EarlyStopping(monitor='val_loss', patience=params_dict['train_params']['early_stop_round'], verbose=1)
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
        callbacks = [checkpoint, earlystop]
    else:
        checkpoint = ModelCheckpoint(model_path, monitor='loss', save_best_only=True, verbose=1)
        callbacks = [checkpoint]
        
    return model, callbacks
