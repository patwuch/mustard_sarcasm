import os
import pandas as pd
import numpy as np
import glob
import torch
from transformers import AutoTokenizer, AutoProcessor, Data2VecTextModel, Data2VecAudioModel
import librosa
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, Data2VecVisionModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''We start off by loading the non_layered data and updating the text embedding quality'''
import pickle
with open('/home/chwu/nonlayered_mustard.pkl', 'rb') as f:
    data = pickle.load(f)
df = pd.DataFrame(data)

'''We create embeddings in this section'''
#Instantiating Data2Vec text model and creating a text-embedding function
from transformers import RobertaTokenizer

from transformers import BartTokenizer, BartModel
import torch

# Load pre-trained BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartModel.from_pretrained('facebook/bart-base')

def get_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    last_four_layers = outputs.last_hidden_state[-4:]  # Extract last four layers
    mean_embeddings = torch.mean(last_four_layers, dim=1)  # Calculate mean of each layer
    return mean_embeddings

#Creating embeddings for the sarcastic utterance
df['text_embeddings'] = df.apply(lambda row: get_embeddings(row['SENTENCE']) if 'u' in row['KEY'] else None, axis=1)

#Creating a for loop that concatenates the various context sentences
#into one string and then converts it into an embedding
prev_scene = None
concatenated_text = ""
separator_token = " </s> "
for index, row in df.iterrows():
    if row['SCENE'] != prev_scene:
        # If new scene, preprocess and store concatenated text
        if prev_scene is not None:
            df.at[prev_index, 'text_embeddings'] = get_embeddings(concatenated_text)
            concatenated_text = ""
        prev_scene = row['SCENE']
        prev_index = index
    
    if 'c' in row['KEY']:
        concatenated_text += row['SENTENCE'] + separator_token

# Preprocess and store last scene's text if any
if concatenated_text:
    df.at[prev_index, 'text_embeddings'] = get_embeddings(concatenated_text)

print("done")
'''Here we dump everything into a pickle.'''

import pickle

# Specify the file path where you want to save the pickled dataframe
file_path = 'nonlayered_mustard_BART_text.pkl'

# Open a file in binary write mode
with open(file_path, 'wb') as f:
    pickle.dump(df, f)
