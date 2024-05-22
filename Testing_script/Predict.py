from Testing_script.TestingDataframes import get_annotation_information
import os
from os import listdir
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import configparser
import sys
import os


current_directory = os.getcwd()

def overlap(start1, end1, start2, end2):
    """how much does the range (start1, end1) overlap with (start2, end2)"""
    return max(max((end2-start1), 0) - max((end2-end1), 0) - max((start2-start1), 0), 0)

folders = ['ModelPredictions', 'Annotations', 'Audio']
paths = {folder: os.path.join(current_directory, 'Testing_script', folder) for folder in folders}

prediction_folder = paths['ModelPredictions']
annotation_folder = paths['Annotations']
audio_folder = paths['Audio']

files=listdir(audio_folder)
predictions=[]
annotations=[]

for file in files:
    file_name=file[0:-4]
    if file_name =='.DS_S':
        continue

    svl=get_annotation_information(audio_folder,annotation_folder,file_name )
    svl["Overlap"]=0
    svl["Cat"]='FN'
    svl.loc[svl.Label=='no-gibbon','Cat']='TN'
    svl["Index"]=np.nan  
    svl["Nb overlap"]=0
    svl['Name']=file_name

    #if svl predicted 
    if file_name+".svl" in listdir(prediction_folder) :
        predict=get_annotation_information(audio_folder,prediction_folder,file_name,True)

        predict["Overlap"]=0
        predict["Cat"]='FP'
        predict["Index"]=np.nan
        predict["Nb overlap"]=0
        predict['Name']=file_name

        #compare predictions vs annotations
        if svl[svl.Label=='roar'].shape[0]!=0:

        for index,row in predict.iterrows() :
            idx=np.abs(np.asarray(svl[svl.Label=='gibbon']['Start']) - row[0]).argmin()  #get the closest window
            lap=overlap(row[0],row[1],svl[svl.Label=='gibbon'].iloc[idx,0],svl[svl.Label=='gibbon'].iloc[idx,1])  #check overlap

            if lap>2 : 
            predict.loc[index,"Overlap"]=lap.copy()
            predict.loc[index,"Cat"]='TP'
            predict.loc[index,"Index"]=idx
            else :
            predict.loc[index,"Overlap"]=lap


        for index,row in predict.iterrows() :
            w=0
            for idx_svl, row_svl in svl[svl.Label=='gibbon'].iterrows():
            lap=overlap(row[0],row[1],row_svl[0],row_svl[1])
            if lap>2 :
                w+=1
            predict.loc[index,"Nb overlap"]=w

        predictions.append(predict)

        #compare annotations vs predictions
        for index,row in svl.iterrows() :
            idx=np.abs(np.asarray(predict['Start']) - row[0]).argmin()  #get the closest window
            lap=overlap(row[0],row[1],predict.iloc[idx,0],predict.iloc[idx,1])  #check overlap

            if ((lap>2) & (svl.loc[index, "Label"]=="gibbon")) : 
            svl.loc[index,"Overlap"]=lap.copy()
            svl.loc[index,"Index"]=idx
            svl.loc[index,"Cat"]='TP'
            elif ((lap>2) & (svl.loc[index, "Label"]=='no-gibbon')):
            svl.loc[index,"Overlap"]=lap.copy()
            svl.loc[index,"Index"]=idx
            svl.loc[index,"Cat"]='FP'
            else :
            svl.loc[index,"Overlap"]=lap 

        for index,row in svl.iterrows() :
            w=0
            for idx_pred, row_pred in predict.iterrows():
            lap=overlap(row[0],row[1],row_pred[0],row_pred[1])
            if lap>2 :
                w+=1
            svl.loc[index,"Nb overlap"]=w

    annotations.append(svl)


Predictions=pd.DataFrame(np.concatenate(predictions, axis=0))
Predictions.columns=predict.columns
Predictions.Index=Predictions.Index.astype(float)

Annotations=pd.DataFrame(np.concatenate(annotations, axis=0))
Annotations.columns=svl.columns
Annotations.Index=Annotations.Index.astype(float)

cat,count=np.unique(Predictions['Cat'], return_counts=True)
print(Predictions['Cat'])
print(cat, count)

cat_a,count_a=np.unique(Annotations['Cat'], return_counts=True)
print(cat_a, count_a)

count_dict = dict(zip(cat_a, count_a))

FP = count_dict.get('FP', 0)  
TN = count_dict.get('TN', 0)  
TP = count_dict.get('TP', 0) 
FN = count_dict.get('FN', 0)  

Precision = TP/(TP + FP)
Recall = TP/(TP + FN)
F_score=TP/(TP+((FN+FP)/2))
Accuracy=(TP+TN)/(TP+TN+FP+FN)


print('Number of calls to detect : ', Annotations[Annotations.Label=='gibbons'].shape[0])
print()
print('False Positives: ', FP)
print('True Positives: ', TP)
print('False Negatives: ', FN)
print('True Negatives: ', TN)
print()
print('F1-score : ', F_score)
print('Accuracy : ', Accuracy)
print('Precision : ', Precision)
print('Recall:',  Recall)
