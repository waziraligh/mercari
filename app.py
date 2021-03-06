# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 20:56:14 2021

@author: Wazir
"""

from flask import Flask, jsonify, request
import numpy as np
from sklearn.externals import joblib
import pandas as pd
#import numpy as np
#from sklearn import linear_model
from sklearn.externals import joblib
from bs4 import BeautifulSoup
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Attention
#import fasttext as ft
from tensorflow.keras.layers import BatchNormalization

import flask
app = Flask(__name__)


''' Defining the function decontracted which expands anything with apostrophes'''
def decontracted(phrase):
   # specific
   phrase = re.sub(r"won\'t", "will not", phrase)
   phrase = re.sub(r"can\'t", "can not", phrase)
   # general
   phrase = re.sub(r"n\'t", " not", phrase)
   phrase = re.sub(r"\'re", " are", phrase)
   phrase = re.sub(r"\'s", " is", phrase)
   phrase = re.sub(r"\'d", " would", phrase)
   phrase = re.sub(r"\'ll", " will", phrase)
   phrase = re.sub(r"\'t", " not", phrase)
   phrase = re.sub(r"\'ve", " have", phrase)
   phrase = re.sub(r"\'m", " am", phrase)
   return phrase
    
''' Defining the function preprocess to preprocess the product name, item_description and brand_name'''
def preprocess(sentence):
   # Converting the sentence to a string instance
   sentence = str(sentence)
   sent = decontracted(sentence)
   sent = sent.replace('\\r',' ')
   sent = sent.replace('\\t',' ')
   sent = sent.replace('\\"',' ')
   sent = sent.replace('\\n',' ')
   sent = re.sub('[^A-Za-z0-9]+',' ',sent)
   sent = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+',' ',sent) # Removing the punctuations
   sent = ' '.join(e for e in sent.split() if e.lower() not in nltk.corpus.stopwords.words('english') and len(e)>=3)
   return sent.lower().strip()

def count_of_words(text):
   '''This function would remove the punctuations, numbers and 
   stopwords from the text and then converting everything to lowercase
   and thereby returning the number of words in the text'''
   try:
       text = text.replace('\\t','') # Removing the tabs
       text = text.replace('\\r','') # Removing the \r
       text = text.replace('\\n','') # Removing the newline character
       text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+','',text) # Removing the punctuations
       text = re.sub('[0-9]+','',text) # Removing the numbers
       # Capturing the clean text in a string and then returning the length of the list for words greater than length of 3        
       new_text = ' '.join(word for word in text.split() if word.lower() not in nltk.corpus.stopwords.words('english') and len(word)>3)
       # Returning the length of the words in the string new_text
       return len(new_text.split())        
   except:
       return 0
    
'''Defining the function to calculate the sentiment score of the preprocessed item description'''
sid = SentimentIntensityAnalyzer()
def sentiment_score(sentence):
   ss = sid.polarity_scores(str(sentence))
   return ss
    
''' Importing the unique training brand list'''
training_brands = pd.read_csv("training_brands.txt",sep = '\t')

'''Initializing the unique training brand list'''
unique_brand_set = list(training_brands.brands.values)

'''Importing the unique general_cat list'''
general_cat = pd.read_csv("gencat.txt", sep='\t')
subcat_1 = pd.read_csv("subcat_1.txt", sep='\t')
subcat_2 = pd.read_csv("subcat_2.txt", sep='\t')

'''Initializing the unique training brand list'''
unique_gencat_set = list(general_cat.General_Category.values)
unique_subcat1_set = list(subcat_1.Subcat_1.values)
unique_subcat2_set = list(subcat_2.Subcat_2.values)

'''Defining the function to replace the brand_name from the unique list of brand names'''
def brand_name_replace(brand_name,product_desc):
  if product_desc.split()[0] in unique_brand_set: # Checking whether the first word of the description is a valid brand
    brand_name = product_desc.split()[0]
  else:
    for i in range(2,9):
        if len(product_desc.split()) >= i and ' '.join(product_desc.split()[0:i]) in unique_brand_set: # Checking whether the first two words of the description is a valid brand
          brand_name = ' '.join(product_desc.split()[0:i])
        break
  return brand_name

def create_model():
  Tokenizer_pritem_desc = joblib.load('Tokenizer_pritem_desc.pkl')
  Tokenizer_brcat = joblib.load('Tokenizer_brcat.pkl')
  scaler = joblib.load('minmaxscaler.pkl')
  '''Loading the embedding matrix'''
  embedding_matrix = np.load('embedding_matrix.npy')
    
  ''' Defining the model variables'''
  maxlen_pritem_desc = 154
  maxlen_brcat = 8
    
  ''' Defining the model architecture'''
  embedding_dim_brcat = 50
  num_tokens_brcat = len(Tokenizer_brcat.word_index)+1
  embedding_dim_preprocessed_pritemdesc = 300
  num_tokens_preprocessed_pritemdesc = len(Tokenizer_pritem_desc.word_index)+1

  # Creating the model architecture
  Inp1 = Input(shape = (maxlen_pritem_desc,), dtype='int64')
  Emb1 = Embedding(input_dim = num_tokens_preprocessed_pritemdesc, output_dim = embedding_dim_preprocessed_pritemdesc, input_length = maxlen_pritem_desc,
                    embeddings_initializer = tf.keras.initializers.constant(embedding_matrix), trainable = False)(Inp1)
  LSTM_layer_1 = LSTM(units=50, return_sequences = True)(Emb1)

  #avgpool = AveragePooling1D(pool_size = 2, strides=2)(Emb1)
  #Flatten_1 = Flatten()(avgpool)

  Inp2 = Input(shape = (maxlen_brcat,), dtype='int64')
  Emb2 = Embedding(input_dim = num_tokens_brcat, output_dim = embedding_dim_brcat, input_length = maxlen_brcat,trainable = True)(Inp2)
  LSTM_layer_2 = LSTM(units=50, return_sequences = True)(Emb2)
  #Flatten_2 = Flatten()(Emb2)

  att_contextvec = Attention()([LSTM_layer_2,LSTM_layer_1])

  avgpool = GlobalAveragePooling1D()(att_contextvec)

  Flatten_1 = Flatten()(avgpool)

  Inp3 = Input(shape = (7,))
  Dense_1 = Dense(units = 4, activation = 'relu', kernel_initializer = 'he_normal')(Inp3)
  #Dense_2 = Dense(units = 4, activation = 'relu', kernel_initializer = 'he_uniform')(Dense_2)

  concat = concatenate([Flatten_1,Dense_1])

  BN_1 = BatchNormalization()(concat)

  Dense_2 = Dense(units = 16, kernel_initializer = 'he_normal')(BN_1)

  Dense_2 = tf.keras.layers.PReLU()(Dense_2)

  dropout_1 = Dropout(0.5)(Dense_2)

  #BN_2 = BatchNormalization()(dropout_1)

  Dense_3 = Dense(units = 8, kernel_initializer = 'he_normal')(dropout_1)

  Dense_3 = tf.keras.layers.PReLU()(Dense_3)

  dropout_2 = Dropout(0.4)(Dense_3)

  Output = Dense(units = 1, activation = 'relu')(dropout_2)

  model = Model(inputs = [Inp1, Inp2, Inp3], outputs = [Output])
  model.load_weights('best_model_weights.hdf5')
  return model,Tokenizer_pritem_desc,Tokenizer_brcat,scaler

model,Tokenizer_pritem_desc,Tokenizer_brcat,scaler = create_model()

@app.route('/')
def hello_world():
  return 'Hello World'

@app.route('/index')
def index():
  return flask.render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
  to_predict_list = request.form.to_dict()
  # Preprocessing the product name
  preprocessed_name = preprocess(to_predict_list["Product_name"])
  # Preprocessing the brand name
  brand_name = to_predict_list["Brand"]
  if brand_name == "":
     brand_name = brand_name_replace(brand_name,to_predict_list["Product_name"])
     # Error Handling for brand_name
     if brand_name == "":
        return jsonify({'Error':'Not a valid brand, please enter a valid brand'})
     else:
        preprocessed_brand_nm = preprocess(brand_name)
  else:
      if brand_name in unique_brand_set:
        preprocessed_brand_nm = preprocess(brand_name)
      else:        
        return jsonify({'Error':'Not a valid brand, please enter a valid brand'})
  
  # Preprocessing the item description
  preprocessed_item_desc = preprocess(to_predict_list["item_description"])
  # Getting the item_description_length of the preprocessed_item_desc
  item_desc_length = count_of_words(preprocessed_item_desc)
  
  general_cat = to_predict_list["General Category"]
  # Preprocessing the general_category
  general_cat = general_cat.replace(" & ","_")
  general_cat = general_cat.replace(" ","_")
  general_cat = general_cat.lower()
  
  if general_cat not in unique_gencat_set:
     return jsonify({'Error':'Not a valid general category. Please enter a valid general category'})
  
  subcat_1 = to_predict_list["Sub-Category_1"]
  # Preprocessing the subcat_1
  subcat_1 = subcat_1.replace(" & ","_")
  subcat_1 = subcat_1.replace(" ","_")
  subcat_1 = subcat_1.replace("-","_")
  subcat_1 = subcat_1.replace("\'s","")
  subcat_1 = subcat_1.replace("\(","")
  subcat_1 = subcat_1.replace("\)","")
  subcat_1 = subcat_1.lower()
  
  if subcat_1 not in unique_subcat1_set:
     return jsonify({'Error':'Not a valid first sub-category. Please enter a valid first sub-category'})
  
  subcat_2 = to_predict_list["Sub-Category_2"]
  # Preprocessing the subcat_2
  subcat_2 = subcat_2.replace(" & ","_")
  subcat_2 = subcat_2.replace(" ","_")
  subcat_2 = subcat_2.replace("-","_")
  subcat_2 = subcat_2.replace("\'s","")
  subcat_2 = subcat_2.replace("\(","")
  subcat_2 = subcat_2.replace("\)","")
  subcat_2 = subcat_2.lower()
  
  if subcat_2 not in unique_subcat2_set:
     return jsonify({'Error':'Not a valid second sub-category. Please enter a valid second sub-category'})  
  
  #Concatenating the preprocessed_name and preprocessed_item_desc
  name_item_desc = preprocessed_name + " " + preprocessed_item_desc
    
  #Concatenating the preprocessed_brand_nm, general_cat, subcat_1 and subcat_2
  concatenated_brcat = preprocessed_brand_nm + " " + general_cat + " " + subcat_1 + " " + subcat_2
    
  #Getting the sentiment score of the preprocessed_item_desc
  sent_sc = sentiment_score(str(preprocessed_item_desc))
  
  #Concatenating the numerical features'''
  numerical_features = [to_predict_list["Item Condition"]] + [to_predict_list["shipping"]] + [item_desc_length] + [v for v in sent_sc.values()]
  
  #Scaling the numerical features
  numerical_features_scaled = scaler.transform(np.array(numerical_features).reshape(1,-1))
    
  #Generating the padded sequence for the concatenated product name and preprocessed_item_desc
  pritem_desc_sequence = Tokenizer_pritem_desc.texts_to_sequences([name_item_desc])
  pritem_desc_padded_sequence = pad_sequences(pritem_desc_sequence, padding = 'post', truncating = 'post', maxlen = 154)
    
  #Generating the padded sequence for the concatenated brand name and the categories
  brcat_sequence = Tokenizer_brcat.texts_to_sequences([concatenated_brcat])
  brcat_padded_sequence = pad_sequences(brcat_sequence, padding = 'post', truncating = 'post', maxlen = 8)
    
  #Predicting on the test instance
  prediction = model.predict([np.array(pritem_desc_padded_sequence).reshape(1,-1), np.array(brcat_padded_sequence).reshape(1,-1), numerical_features_scaled.reshape(1,-1)])
  
  return jsonify({'price':str(prediction[0][0])})
  
if __name__=='__main__':
    app.run(host='0.0.0.0', port=5050)
    
    