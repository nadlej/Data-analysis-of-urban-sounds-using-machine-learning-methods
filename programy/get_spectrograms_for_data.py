import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import seaborn as sn
import warnings
import time 
import csv

warnings.filterwarnings('ignore')

df = pd.read_csv('../UrbanSound8K.csv')
types_of_sounds = ['oryginal', 'bgnoise', 'drc', 'pitch1', 'pitch2', 'stretch']

# Inicjalizacja podstawowych parametrów 
HOP_LENGTH = 0.023
WINDOW_LENGTH = 0.023    
N_MEL = 128            
SOUND_DURATION = 3.0

def get_mels(file_path):
  audio, sample_rate = librosa.load(file_path, duration=SOUND_DURATION, res_type='kaiser_fast')
  mels = compute_melspectrogram_with_fixed_length(audio, sample_rate)
  return mels

def compute_melspectrogram_with_fixed_length(audio, sampling_rate, num_of_samples=128):
    try:
        # Uzyskanie spektrogramu w skali melowej 
        melspectrogram = librosa.feature.melspectrogram(y=audio, 
                                                        sr=sampling_rate, 
                                                        hop_length=int(HOP_LENGTH*sampling_rate),
                                                        win_length=int(WINDOW_LENGTH*sampling_rate), 
                                                        n_mels=N_MEL)

        # Przekonwertowanie skali mocy do skali decybelowej 
        melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)
        
        # Zapisanie dlugosci spektrogramu
        melspectrogram_length = melspectrogram_db.shape[1]
        
        # Sprawdzenie długosci spektrogramu i podjecie decyzji od niej zaleznych 
        if melspectrogram_length != num_of_samples:
            melspectrogram_db = librosa.util.fix_length(melspectrogram_db, 
                                                        size=num_of_samples, 
                                                        axis=1, 
                                                        constant_values=(0, -80.0))
    except Exception as e:
        print("\nError\n>>", e)
        return None 
    
    return melspectrogram_db

# Tablica w ktorej beda przechowywane wszystkie spektrogramy z danymi
features = []

for i in range(8732):
     oryginal_file_name = df["slice_file_name"][i]
     folder_id = str(df["fold"][i])
     audio_class_id = df["classID"][i]
     
     # Uzyskanie sciezki do pliku audio
     file_name = '../fold' + folder_id + '/' + types_of_sounds[0] + '/audio/' + oryginal_file_name

     # Obliczenie spektrogramu przeskalowanego w skali melowej
     mels = get_mels(file_name)
     
     features.append([mels, audio_class_id, folder_id]) 
    
     print(str(i+1) + "/8732")

us8k_df = pd.DataFrame(features, columns=["melspectrogram", "audio_class_id", "folder_id"])
us8k_df.to_pickle("us8k_df.pkl")