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

current_directory = os.getcwd()

def prepare_data(data_path):
    """Load and prepare data"""
    df = pd.read_pickle(data_path)
    df['index'] = df['index'].map({'gibbon': 1, 'no-gibbon': 0})
    class_counts = df['index'].value_counts()
    min_count = min(class_counts)
    df = df.groupby('index').apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    df = shuffle(df)
    
    y_train = tf.one_hot(df.iloc[:, 0], 2)
    y_train = np.array(y_train).astype("float32")
    
    x_train = np.array(df.iloc[:, 1:]).reshape(df.shape[0], 128, 128)
    x_train = np.expand_dims(x_train, -1).astype("float32")
    x_train = tf.repeat(x_train, 3, axis=-1)
    
    print("Shape of x_train:", x_train.shape)
    print()
    print("Shape of y_train:", y_train.shape)
      
    return x_train, y_train

data_path = os.path.join(current_directory, 'data', 'gibbon_processed.pkl')
x_train, y_train = prepare_data(data_path)

checkpoint_path = os.path.join(current_directory, 'cnn_checkpoint', 'checkpoint', 'cp.ckpt')

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, monitor='loss', save_weights_only=True, mode='max', verbose=1
)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

networks = CNNNetwork()
model = networks.resnet()

model.fit(
    x=x_train, y=y_train, batch_size=16, epochs=20, verbose=1, 
    callbacks=[checkpoint_callback, early_stopping_callback]
)

predict = PredictionHelper(True)
predict.predict_all_test_files(True)



        
        
        
