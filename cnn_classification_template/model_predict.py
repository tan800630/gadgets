from argparse import ArgumentParser

# parameter set

parser = ArgumentParser()
parser.add_argument("--config_path", help="path of configuration")
parser.add_argument("--out_path", help="path for saving prediction")
parser.add_argument("--batch_size", help="batch size for predict")
parser.add_argument("--img_path", help="path for testing data (images)")

args = parser.parse_args()

config_path = args.config_path
out_path = args.out_path
batch_size = args.batch_size
img_path = args.img_path

from cnn_utils import *

#########################################################




def main():
    
    print('--Start--')
    
    # load configuration
    params_dict = json.load(open(config_path,'r'))
    
    preproc = get_preproc_function(scale_range = params_dict['train_params']['scale_range'],
                                   histogram_equalization = params_dict['train_params']['histogram_equalization'])

    model = load_model(os.path.join(params_dict['model_params']['model_dir'], params_dict['model_params']['model_name']+'.h5'))
    
    test_gen = generator_from_dir(img_path,
                              target_size = tuple(params_dict['train_params']['img_shape'][:2]),
                              batch_size = batch_size,
                              preprocess_function = preproc)

    test_prediction = []
    test_label = []
    file_list = []
    
    # testing data prediction
    for i, [image_batch, file_batch] in enumerate(test_gen):
        prediction = model.predict(np.array(image_batch))

        test_prediction.extend(prediction)
        file_list.extend(file_batch)
        if i%10==0:
            print('{} batchs had been processed.'.format(i))

    # 存出檔案
    out_file = pd.DataFrame({
        'file_name' : file_list,
        'prediction' : np.argmax(np.array(test_prediction), axis = 1),
    })

    out_file.to_csv(os.path.join(out_path,'predict.csv'), index = False)
    
    print('--Finish--')

if __name__ == "__main__":
    main()