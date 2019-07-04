from argparse import ArgumentParser

# parameter set

parser = ArgumentParser()
parser.add_argument("--config_path", help="path of configuration")
parser.add_argument("--log_path", help="path for saving log")
args = parser.parse_args()

config_path = args.config_path
log_path = args.log_path

from cnn_utils import *

#########################################################


def main():
    
    print('--Start--')

    
    # load configuration
    params_dict = json.load(open(config_path,'r'))
    
    dat_dict = get_and_save_data(params_dict)
    
    preproc = get_preproc_function(scale_range = params_dict['train_params']['scale_range'],
                                   histogram_equalization = params_dict['train_params']['histogram_equalization'])

    train_gen = image_data_generator(dat_dict['training_set'], img_dir = params_dict['data_params']['img_dir'],
                                     file_col = params_dict['data_params']['file_col'],
                                     target_col = params_dict['data_params']['target_col'],
                                     batch_size = params_dict['train_params']['batch_size'],
                                     re_size = params_dict['train_params']['img_shape'][0:2],
                                     preprocess_function=preproc,
                                     augmentation = params_dict['train_params']['img_augmentation'])
    if isinstance(dat_dict['validation_set'], pd.DataFrame):
        valid_gen = image_data_generator(dat_dict['validation_set'], img_dir = params_dict['data_params']['img_dir'],
                                         file_col = params_dict['data_params']['file_col'],
                                         target_col = params_dict['data_params']['target_col'],
                                         batch_size = params_dict['train_params']['batch_size'],
                                         re_size = params_dict['train_params']['img_shape'][0:2],
                                         preprocess_function=preproc,
                                         augmentation = False)
    
    n_class = dat_dict['training_set'][params_dict['data_params']['target_col']].nunique()
    
    model, callbacks = build_model(params_dict, n_class)
    if isinstance(dat_dict['validation_set'], pd.DataFrame):
        model_history = model.fit_generator(train_gen,
                                            epochs = params_dict['train_params']['epochs'],
                                            validation_data = valid_gen,
                                            callbacks = callbacks,
                                            verbose=1,
                                            steps_per_epoch = math.ceil(len(dat_dict['training_set'])/params_dict['train_params']['batch_size']),
                                            validation_steps = math.ceil(len(dat_dict['validation_set'])/params_dict['train_params']['batch_size']))
    else:
        model_history = model.fit_generator(train_gen, epochs = params_dict['train_params']['epochs'],
                                            callbacks = callbacks,
                                            verbose = 1,
                                            steps_per_epoch = math.ceil(len(dat_dict['training_set'])/params_dict['train_params']['batch_size']))
    
    print('End of training')
    
    print('Start to predict data and save logs. may take a while....')
    log_dict = {}
    log_dict['general'] = {}
    log_dict['general']['model_path'] = os.path.join(params_dict['model_params']['model_dir'], params_dict['model_params']['model_name']+'.h5')
    log_dict['general']['config_path'] = config_path
    
    model = load_model(log_dict['general']['model_path'])
    
    for dat_type, dat in dat_dict.items():

        log_dict[dat_type] = {}

        gen = image_data_generator(dat, img_dir = params_dict['data_params']['img_dir'],
                                         file_col = params_dict['data_params']['file_col'],
                                         target_col = params_dict['data_params']['target_col'],
                                         batch_size = params_dict['train_params']['batch_size'],
                                         re_size = params_dict['train_params']['img_shape'][0:2],
                                         preprocess_function=preproc,
                                         augmentation = False, shuffle_ = False)

        true_ = np.argmax(pd.get_dummies(dat[params_dict['data_params']['target_col']]).values, axis = 1)
        pred = model.predict_generator(gen, steps = math.ceil(len(dat)/params_dict['train_params']['batch_size']))

        cm = confusion_matrix(y_true = true_, y_pred = np.argmax(pred, axis = 1))
        acc = accuracy_score(y_true = true_, y_pred = np.argmax(pred, axis = 1))
        precision = precision_score(y_true = true_, y_pred = np.argmax(pred, axis = 1))
        recall = recall_score(y_true = true_, y_pred = np.argmax(pred, axis = 1))

        # save result to log_dict
        log_dict[dat_type]['length'] = dat.shape[0]
        log_dict[dat_type]['accuracy'] = acc
        log_dict[dat_type]['confusion_matrix'] = cm.tolist()
        log_dict[dat_type]['precision'] = precision
        log_dict[dat_type]['recall'] = recall
    
    log_name = os.path.join(log_path,params_dict['model_params']['model_name']+'.log')
    
    with open(log_name, 'w') as f:
        f.write(json.dumps(log_dict, indent=3))
    
    print('--Finish--')

if __name__ == "__main__":
    main()