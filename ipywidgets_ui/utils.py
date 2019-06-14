
import pandas as pd
import numpy as np
import cv2
import os
import pickle
import math

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from IPython.display import display, HTML
from ipywidgets import interact, interactive, Layout
import ipywidgets as widgets

import matplotlib.pyplot as plt


def button_after_click(b, ui):
    b.button_style = ''
    b.disabled = True
    b.description = 'Done'
    
    # disable ipywidgets elements
    for ch in ui.children:
        for ch_ch in ch.children:
            
            try:
                for ch_ch_ch in ch_ch.children:
                    ch_ch_ch.disabled = True
            except:
                pass
            ch_ch.disabled = True
            

def set_configuration(config_):
    
    def setting_accord():
        general_box = widgets.Box(children=[widgets.IntSlider(value = 2, min = 1, max = 8, description = '# of GPUs:')])

        path_box = widgets.Box(children=[widgets.Text(value = 'example: ~/project_dir/', description='Main  Directory:',
                                                      style = {'description_width': 'initial'}, layout = Layout(flex='3 1 0%', width='auto'))], layout=Layout(display='flex', flex_flow='column', align_items='stretch', width='50%'))

        # total accordion
        tab_accord = widgets.Accordion(selected_index=None)
        tab_accord.children = [general_box, path_box]
        tab_accord.set_title(0, 'General setting')
        tab_accord.set_title(1, 'Path')

        return tab_accord

    def get_config(b):

        # general setting
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in range(tab_accord.children[0].children[0].value)])

        # directory
        config_['main_dir'] = tab_accord.children[1].children[0].value
        config_['random_state'] = 1048
        
        button_after_click(b, tab_accord)
    
    title = HTML('<h1>Configuration</h1>')
    tab_accord = setting_accord()
    b = widgets.Button(description = 'Set Config', button_style = 'primary')
    display(title)
    display(tab_accord)
    display(b)
    
    b.on_click(get_config)
    

def set_data_configuration(data_params_):
    
    def setting_accord():
        data_box = widgets.Box([widgets.Text(value = 'example: data.csv', description='Data Path:', style = {'description_width': 'initial'}, layout = Layout(flex='3 1 0%', width='auto')), 
                                widgets.Text(value = 'example: ~/images/', description='Image Directory:', style = {'description_width': 'initial'}, layout = Layout(flex='3 1 0%', width='auto')),
                                widgets.Text(value = '', description='Colname of Filename:', style = {'description_width': 'initial'}),
                                widgets.Text(value = '', description='Colname of Label:', style = {'description_width': 'initial'})
                               ], layout=Layout(display='flex', flex_flow='column', align_items='stretch', width='80%'))


        preprocess_box = widgets.Box([widgets.HBox([widgets.Checkbox(value=False, description='Save split data',disabled=False),
                                                    widgets.Checkbox(value = True, description = 'Image Augmentation', disabled = False),
                                                    widgets.Checkbox(value = False, description = 'Histogram Equalization', disabled = False)]),
                                      widgets.RadioButtons(options = [(224, 224, 3), (299, 299, 3)], description = 'resize:'),

                                      widgets.FloatRangeSlider(value = (-1.0, 1.0), min = -1.0, max = 1.0, description = 'Range of Normalization:',
                                                                style = {'description_width': 'initial'}),
                                      widgets.FloatSlider(value = 0.2, min = 0.1, max = 1.0, description = 'Validation Set:',
                                                           style = {'description_width': 'initial'})
                                     ], layout=Layout(display='flex', flex_flow='column', align_items='stretch', width='80%'))

        tab_accord = widgets.Accordion(selected_index=None)
        tab_accord.children = [data_box, preprocess_box]
        tab_accord.set_title(0, 'Dataset')
        tab_accord.set_title(1, 'Preprocess')

        return tab_accord    

    def get_data_params(b):

        # params of Dataset
        data_params_['data_path'] = tab_accord.children[0].children[0].value
        data_params_['img_dir'] = tab_accord.children[0].children[1].value
        data_params_['file_name'] = tab_accord.children[0].children[2].value
        data_params_['label_name'] = tab_accord.children[0].children[3].value

        # params for preprocess
        data_params_['save_dataset'] = tab_accord.children[1].children[0].children[0].value
        data_params_['augmentation'] = tab_accord.children[1].children[0].children[1].value
        data_params_['he'] = tab_accord.children[1].children[0].children[2].value
        data_params_['re_size'] = tab_accord.children[1].children[1].value
        data_params_['normalization_range'] = tab_accord.children[1].children[2].value

        button_after_click(b, tab_accord)

    title = HTML('<h1>Data setting</h1>')
    tab_accord = setting_accord()
    b = widgets.Button(description = 'Set df Params', button_style = 'primary')
    display(title)
    display(tab_accord)
    display(b)
    
    b.on_click(get_data_params)


def set_model_configuration(model_params_):
    
    def setting_accord():
        model_box = widgets.Box([widgets.RadioButtons(options = ['ImageNet pretrain model', 'Custom pretrain model'], description = ''),
                                 widgets.Dropdown(options=['ResNet50', 'Xception', 'DenseNet121'], value='ResNet50', description='Model Architecture:', disabled=False, style = {'description_width': 'initial'}),
                                 widgets.Text(value = 'example: ./mymodel.h5', description='Model Path:', style = {'description_width': 'initial'}, layout = Layout(flex='3 1 0%', width='auto'), disabled = True),
                                ], layout=Layout(display='flex', flex_flow='column', align_items='stretch', width='80%'))

        params_box = widgets.Box(children=[widgets.Dropdown(options=['8', '16', '32', '64', '128'], description='Batch size:'),
                                           widgets.FloatText(value = 10, min = 1, max = 200, description='Epochs:'),
                                           widgets.FloatText(value = 5, min = 2, max = 10, description='Early stopping rounds:',
                                                             style = {'description_width': 'initial'}),
                                           widgets.FloatLogSlider(value = -5, min = -7, max = -1, step = 1,
                                                                  description='Learning rate:', readout_format = '.1e'),
                                           widgets.FloatSlider(value = 0.5, min = 0.0, max = 0.8, step = 0.1,
                                                               description='Drop-out Rate:', style = {'description_width': 'initial'}),
                                                 ], layout=Layout(display='flex', flex_flow='column', align_items='stretch', width='50%'))

        model_save_box = widgets.Box(children=[widgets.Text(value = 'example: ./saved_models/', description='Saved Path:',
                                                            style = {'description_width': 'initial'}, layout = Layout(flex='3 1 0%', width='auto')),
                                               widgets.Text(value = 'example: mymodel.h5', description='Model Name:',
                                                            style = {'description_width': 'initial'}, layout = Layout(flex='3 1 0%', width='auto'))
                                              ], layout=Layout(display='flex', flex_flow='column', align_items='stretch', width='50%'))

        tab_accord = widgets.Accordion(selected_index=None)
        tab_accord.children = [model_box, params_box, model_save_box]
        tab_accord.set_title(0, 'Model Architecture')
        tab_accord.set_title(1, 'Training Parameters')
        tab_accord.set_title(2, 'Save Settings')
    
        return tab_accord

    def get_model_params(b):

        # model architecture
        model_params_['model_type'] = tab_accord.children[0].children[0].value
        model_params_['model'] = tab_accord.children[0].children[1].value
        model_params_['custom_model_path'] = tab_accord.children[0].children[2].value

        # training params
        model_params_['batch_size'] = int(tab_accord.children[1].children[0].value)
        model_params_['epochs'] = tab_accord.children[1].children[1].value
        model_params_['early_stop_rounds'] = tab_accord.children[1].children[2].value
        model_params_['lr'] = tab_accord.children[1].children[3].value
        model_params_['dr_rate'] = tab_accord.children[1].children[4].value

        # save setting
        model_params_['save_dir'] = tab_accord.children[2].children[0].value
        model_params_['model_name'] = tab_accord.children[2].children[1].value

        button_after_click(b, tab_accord)

    title = HTML('<h1>Model Setting</h1>')
    tab_accord = setting_accord()
    
    
    def obs(b): 
        if tab_accord.children[0].children[0].value=='ImageNet pretrain model':
            tab_accord.children[0].children[1].disabled = False
            tab_accord.children[0].children[2].disabled = True
        else:
            tab_accord.children[0].children[1].disabled = True
            tab_accord.children[0].children[2].disabled = False
    
    b = widgets.Button(description = 'Set model Params', button_style = 'primary')
    display(title)
    display(tab_accord)
    display(b)

    tab_accord.children[0].children[0].observe(obs)
    b.on_click(get_model_params)


def set_test_configuration(test_params_):
    
    def setting_accord():
        data_box = widgets.Box([widgets.Text(value = 'example: test_dat.csv', description='Data Path:',
                                                          style = {'description_width': 'initial'}, layout = Layout(flex='3 1 0%', width='auto')),
                                widgets.Text(value = 'example: ~/images/', description='Image Directory:',
                                                          style = {'description_width': 'initial'}, layout = Layout(flex='3 1 0%', width='auto')),
                                widgets.Text(value = '', description='Colname of Filename:', style = {'description_width': 'initial'}),
                                widgets.Text(value = '', description='Colname of Label:', style = {'description_width': 'initial'})
                               ], layout=Layout(display='flex', flex_flow='column', align_items='stretch', width='80%'))


        preprocess_box = widgets.Box([widgets.HBox([widgets.Checkbox(value=False, description='Save split data',disabled=False),
                                                     widgets.Checkbox(value = True, description = 'Image Augmentation', disabled = False)]),
                                      widgets.RadioButtons(options = [(224, 224, 3), (299, 299, 3)], description = 'resize:'),

                                      widgets.FloatRangeSlider(value = (-1.0, 1.0), min = -1.0, max = 1.0, description = 'Range of Normalization:',
                                                                style = {'description_width': 'initial'}),
                                      widgets.FloatSlider(value = 0.2, min = 0.1, max = 1.0, description = 'Validation Set:',
                                                           style = {'description_width': 'initial'})
                                     ], layout=Layout(display='flex', flex_flow='column', align_items='stretch', width='80%'))

        model_box = widgets.Box([widgets.Text(value = 'example: ./mymodel.h5', description='Model Path:',
                                                            style = {'description_width': 'initial'}, layout = Layout(flex='3 1 0%', width='auto')),
                                 widgets.Dropdown(options=['8', '16', '32', '64', '128'], description='Batch size:'),
                                 widgets.Text(value = '', description='Prediction file path:',
                                                            style = {'description_width': 'initial'}, layout = Layout(flex='3 1 0%', width='auto')),
                                ], layout=Layout(display='flex', flex_flow='column', align_items='stretch', width='50%'))

        tab_accord = widgets.Accordion(selected_index=None)
        tab_accord.children = [data_box, preprocess_box, model_box]
        tab_accord.set_title(0, 'Dataset')
        tab_accord.set_title(1, 'Preprocess')
        tab_accord.set_title(2, 'Model')

        return tab_accord

    def get_test_params(b):

        # params of Dataset
        test_params_['data_path'] = tab_accord.children[0].children[0].value
        test_params_['img_dir'] = tab_accord.children[0].children[1].value
        test_params_['file_name'] = tab_accord.children[0].children[2].value
        test_params_['label_name'] = tab_accord.children[0].children[3].value
        
        # params for preprocess
        test_params_['save_dataset'] = tab_accord.children[1].children[0].children[0].value
        test_params_['augmentation'] = tab_accord.children[1].children[0].children[1].value
        test_params_['re_size'] = tab_accord.children[1].children[1].value
        test_params_['normalization_range'] = tab_accord.children[1].children[2].value

        # params for testing
        test_params_['Model path'] = tab_accord.children[2].children[0].value
        test_params_['batch_size'] = int(tab_accord.children[2].children[1].value)
        test_params_['out_path'] = tab_accord.children[2].children[2].value

        button_after_click(b, tab_accord)
    
    title = HTML('<h1>Test Setting</h1>')
    tab_accord = setting_accord()
    b = widgets.Button(description = 'Set Params', button_style = 'primary')
    display(title)
    display(tab_accord)
    display(b)
    
    b.on_click(get_test_params)

def training_plot(model_history):
    training_loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.plot(training_loss, label="training_loss")
    plt.plot(val_loss, label="validation_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend(loc='best')
    plt.show()

###################################################################

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
            img = cv2.resize(img, re_size)

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

def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def generate_preproc_func(range, he = False):
    
    if he:
        def preproc(x):
            x = hisEqulColor(x)
            return (x / 255.0)*(range[1]-range[0])+ range[0]
    else:
        def preproc(x):
            return (x / 255.0)*(range[1]-range[0])+ range[0]
    return preproc

def take_test_id(data, random_state , test_size = 0.2, target = 'View Position'):
    d = shuffle(data)
    test_len = len(data)*test_size
    test_index = d.groupby(target).head(int(test_len/2)).index
    return test_index


def load_data(config_, data_params_):
    
    data = pd.read_csv(data_params_['data_path'])
    t_dat, v_dat = train_test_split(data, shuffle = True, test_size = 0.2, random_state = config_['random_state'])
    
    if data_params_['save_dataset']:
        pickle.dump({'train_dat':t_dat, 'valid_dat':v_dat}, open('split_data.pkl', 'wb'))
        print('Save file to \'split_data.pkl\'')
    return [t_dat, v_dat]

def preproc_generator(data, data_params_, model_params_):
    
    preproc = generate_preproc_func(data_params_['normalization_range'], data_params_['histogram_equalization'])


    train_gen=image_data_generator(df = data[0],
                                 img_dir = data_params_['img_dir'],
                                 file_col = data_params_['file_name'],
                                 target_col = data_params_['label_name'],
                                 batch_size = model_params_['batch_size'],
                                 re_size = data_params_['re_size'][0:2],
                                 preprocess_function = preproc,
                                 augmentation = data_params_['augmentation'],
                                 shuffle_ = True)

    valid_gen=image_data_generator(df = data[1],
                                 img_dir = data_params_['img_dir'],
                                 file_col = data_params_['file_name'],
                                 target_col = data_params_['label_name'],
                                 batch_size = model_params_['batch_size'],
                                 re_size = data_params_['re_size'][0:2],
                                 preprocess_function = preproc,
                                 augmentation = data_params_['augmentation'],
                                 shuffle_ = False)
    
    return [train_gen, valid_gen]

def build_model(data_params_, model_params_):
    
    if model_params['model_type'] == 'ImageNet pretrain model':
        if model_params_['model'] == 'ResNet50':
            base_model = ResNet50(weights = 'imagenet', include_top = False, input_shape = data_params_['re_size'])
        elif model_params_['model'] == 'Xception':
            base_model = Xception(weights = 'imagenet', include_top = False, input_shape = data_params_['re_size'])
        elif model_params_['model'] == 'DenseNet121':
            base_model = DenseNet121(weights = 'imagenet', include_top = False, input_shape = data_params_['re_size'])

        model = base_model.output
        model = Flatten()(model)
        model = Dropout(model_params_['dr_rate'])(model)
        predictions = Dense(2, activation = 'softmax')(model) #class
        model = Model(inputs = base_model.input, outputs = predictions)
    elif model_params['model_type'] == 'Custom pretrain model':
        model = load_model(model_params_['custom_model_path'])
    
    optimizer = keras.optimizers.Adam(lr = model_params_['lr'])
    
    save_path = os.path.join(model_params_['save_dir'], model_params_['model_name']+'.h5')
    checkpoint = ModelCheckpoint(save_path, monitor = 'val_loss', save_best_only = True, verbose = 1)
    early_stop = EarlyStopping(monitor = 'val_loss', patience = model_params_['early_stop_rounds'], verbose = 1)
    
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    
    return model, early_stop, checkpoint

def training_(config_, df_params_, model_params_):
    [t_dat, v_dat] = load_data(config_, df_params_)
    [train_gen, valid_gen] = preproc_generator([t_dat, v_dat], df_params_, model_params_)
    model, early_stop, checkpoint = build_model(df_params_, model_params_)
    
    model_history = model.fit_generator(train_gen,
                                        epochs = model_params_['epochs'],
                                        validation_data = valid_gen, 
                                        callbacks = [early_stop],  #, checkpoint
                                        verbose = 1, 
                                        steps_per_epoch = int(len(t_dat)/model_params_['batch_size']),
                                        validation_steps = int(len(v_dat)/model_params_['batch_size']))
    return model, model_history


def model_testing(test_params_):
    
    model = load_model(test_params_['Model path'])
    test_dat = pd.read_csv(test_params_['data_path'])
    
    preproc = generate_preproc_func(test_params_['normalization_range'])
    
    # test data generator
    test_gen=image_data_generator(df = test_dat,
                                 img_dir = test_params_['img_dir'],
                                 file_col = test_params_['file_name'],
                                 target_col = test_params_['label_name'],
                                 batch_size = test_params_['batch_size'],
                                 re_size = test_params_['re_size'][0:2],
                                 preprocess_function = preproc,
                                 augmentation = test_params_['augmentation'],
                                 shuffle_ = False)


    test_prediction = []
    test_label = []

    # testing data prediction
    max_iter = math.ceil(len(test_dat)/test_params_['batch_size'])
    bar = widgets.IntProgress(value = 0, min = 0, max = max_iter)
    display(bar)
    
    for i in range(max_iter):
        [image_out, label_out] =next(test_gen)
        test_prediction.extend(model.predict(image_out))
        test_label.extend(label_out)
        
        bar.value = i

    test_prediction = np.array(test_prediction)
    test_label = np.array(test_label)
    file_name = test_dat['Image Index']
    
    pd.DataFrame({'file name':file_name, 'prediction':test_prediction}).to_csv(test_params_['out_path'], index = False)
    print('Save output file :{}'.format(test_params_['out_path']))
    
    return test_prediction, test_label