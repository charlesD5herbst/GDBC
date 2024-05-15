from itertools import product
from results.metrics import mean_metrics, std_metrics, config_var
from results.data_saver import save_results
import matplotlib.pyplot as plt
from statistics import mean
from statistics import stdev
from models.CNNNetwork import *
from Testing_script.PredictionHelper import *
from Testing_script.Predict import compute_metrics
import time
import configparser
import sys
import os
from sklearn.utils import shuffle
import csv


config_file_path = sys.argv[1]
base_name, extension = os.path.splitext(config_file_path)
print(f"Config file name (without extension): {base_name}")

config = configparser.ConfigParser()
config.read(config_file_path)

current_directory = os.getcwd()

for section in config.sections():
    
    cnn_model = config[section]['select_model']
    print(cnn_model)
    
    temp_metrics = {
        'accuracy': [] ,
        'f1_score': [] ,
        'precision': [] ,
        'recall': []
    }

    df = pd.read_pickle(current_directory+"/data/all_lemur_data.pkl")
    print(df.head(10))
    df['index'] = df['index'].map({'roar': 1 , 'no-roar': 0})
    print(df)
    class_counts = df['index'].value_counts()
    min_count = min(class_counts)
    df = df.groupby('index').apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    df = shuffle(df)
    print(df)
   
    for _ in range(10):

        y_train = df.iloc[:, 0]
        y_train = tf.one_hot(y_train, 2)
        y_train = np.array(y_train).astype("float32")
        x_train = df.iloc[:, 1::]
        x_train = np.array(x_train).reshape(x_train.shape[0], 128, 128)
        x_train = np.expand_dims(x_train, -1).astype("float32")
        x_train = x_train.reshape(-1, 128, 128, 1)
        print(x_train.shape)
        print(y_train.shape)

        networks = CNNNetwork()
        
        model = networks.custom_model() if cnn_model == 'custom_cnn' else networks.resnet()
        if cnn_model != 'custom_cnn':
            x_train = tf.expand_dims(x_train, axis=3, name=None)
            x_train = tf.repeat(x_train, 3, axis=3)


        checkpoint_path = current_directory+f'/cnn_checkpoint/checkpoint/{base_name}/cp.ckpt'
        callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path , monitor = 'loss', save_weights_only=True , mode = 'max', verbose=1)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
        
        model.fit(x=x_train, y=y_train, batch_size=16, epochs=5, verbose=1, callbacks = [callback, early_stopping_callback])
        

        predict = PredictionHelper(cnn_model)
        predict.predict_all_test_files(True)
        accuracy, precision, recall, f_score = compute_metrics()

        temp_metrics['accuracy'].append(accuracy)
        temp_metrics['f1_score'].append(f_score)
        temp_metrics['precision'].append(precision)
        temp_metrics['recall'].append(recall)
        

    mean_metrics['accuracy'].append(mean(temp_metrics['accuracy']))
    mean_metrics['f1_score'].append(mean(temp_metrics['f1_score']))
    mean_metrics['precision'].append(mean(temp_metrics['precision']))
    mean_metrics['recall'].append(mean(temp_metrics['recall']))

    std_metrics['accuracy'].append(stdev(temp_metrics['accuracy']))
    std_metrics['f1_score'].append(stdev(temp_metrics['f1_score']))
    std_metrics['precision'].append(stdev(temp_metrics['precision']))
    std_metrics['recall'].append(stdev(temp_metrics['recall']))

    config_var['cnn'].append(cnn_model)
    config_var['num_training_examples'].append(num_training_examples)
    config_var['num_to_generate'].append(num_to_generate)

#df = save_results(mean_metrics, std_metrics, config_var, base_name)
print(df)



        
        
        