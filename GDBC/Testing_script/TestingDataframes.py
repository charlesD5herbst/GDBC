import glob, os
import ntpath
from os import listdir

import numpy as np
import random
import pandas as pd
import time
import librosa.display
import librosa
from scipy import signal
import soundfile as sf
import datetime

from yattag import Doc, indent
import ntpath

import time
from xml.dom import minidom
from datetime import datetime
from datetime import datetime, timedelta

import glob, os
import ntpath
from os import listdir

import numpy as np
import random
import pandas as pd
import time
import librosa.display
import librosa
from scipy import signal
import soundfile as sf
import datetime

from yattag import Doc, indent
import ntpath

import time
from xml.dom import minidom
import xml.etree.ElementTree as ET
from datetime import datetime
from datetime import datetime, timedelta



def read_audio_file(file_name, species_folder):
    '''
    file_name: string, name of file including extension, e.g. "audio1.wav"

    '''
    # Get the path to the file
    audio_folder = os.path.join(species_folder,file_name)

    # Read the amplitudes and sample rate
    audio_amps, audio_sample_rate = librosa.load(audio_folder, sr=None)

    return audio_amps, audio_sample_rate

def get_annotation_information(audio_folder, annotation_folder,file_name, predicted=False):


    if predicted==True:
        file_name_annotation=file_name+".svl"
    else:
        file_name_annotation=file_name+".svl"
    # Process the .svl xml file
    print("here",annotation_folder+file_name_annotation)
    xmldoc = minidom.parse(annotation_folder+file_name_annotation)
    
    itemlist = xmldoc.getElementsByTagName('point')
    idlist = xmldoc.getElementsByTagName('model')

    start_time = []
    end_time = []
    labels = []
    audio_file_name = ''

    if (len(itemlist) > 0):

        file_name_no_extension = file_name
        print (file_name_no_extension)
        audio_amps, original_sample_rate = read_audio_file(file_name+".wav", audio_folder)     
        # Iterate over each annotation in the .svl file (annotatation file)
        for s in itemlist:

            # Get the starting seconds from the annotation file. Must be an integer
            # so that the correct frame from the waveform can be extracted
            start_seconds = float(s.attributes['frame'].value)/original_sample_rate
                    
            # Get the label from the annotation file
            label = str(s.attributes['label'].value)

            # Set the default confidence to 10 (i.e. high confidence that
            # the label is correct). Annotations that do not have the idea
            # of 'confidence' are teated like normal annotations and it is
            # assumed that the annotation is correct (by the annotator). 
            label_confidence = 10

            # Check if a confidence has been assigned
            if ',' in label:

                # Extract the raw label
                lalel_string = label[:label.find(','):]

                # Extract confidence value
                label_confidence = int(label[label.find(',')+1:])

                # Set the label to the raw label
                label = lalel_string


            # If a file has a blank label then skip this annotation
            # to avoid mislabelling data
            if label == '':
                break

            # Only considered cases where the labels are very confident
            # 10 = very confident, 5 = medium, 1 = unsure this is represented
            # as "SPECIES:10", "SPECIES:5" when annotating.
            if label_confidence == 10:
                # Get the duration from the annotation file
                annotation_duration_seconds = float(s.attributes['duration'].value)/original_sample_rate
                start_time.append(start_seconds)
                end_time.append(start_seconds+annotation_duration_seconds)
                labels.append(label)

    df_svl_gibbons = pd.DataFrame({'Start': start_time, 'End':end_time ,'Label': labels})
    return df_svl_gibbons