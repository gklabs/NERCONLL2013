
'''
NER.py
AIT 726 Assignment 3: Named Entity Recognition
Haritha G, Giridhar K
Spring 2020


Instructions for execution



Pseudocode
1. Read data
	CONLL2013
2.Preliminary analysis
	Plot POS frequency
	Plot NE frequency
	Word cloud

3. Formatting
    Lower case capitalized words
	Stop words are not removed
	No Stemming
	No Lemmatization
	Tokenization is already done
	padding 0s and tag is  <pad>
4. Vector Representations
	pretrained word embeddings given (word2vec) 300 dim x 3M words


5. Classification architechtures Training
	RNN
		one layer 256 hidden units
		Fully connected output layer
		Adam optimizer
		cross-entropy 
		alpha 0.0001
		2000 minibatches / epoch
	Bidirectional RNN
		one layer 256 hidden units
		Fully connected output layer
		Adam optimizer
		cross-entropy 
		alpha 0.0001
		2000 minibatches / epoch



6. Performance Evaluation
	Parameter Tuning for each of the models
		Validation accuracy
	Test Accuracy 
	Confusion Matrix

	'''
import os
import sys
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from pathlib import Path

import nltk as nl
from nltk.tokenize import word_tokenize
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer

import torch
import torch.optim as optim
from torch.autograd import Variable
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


'''
This method does the following:
    1. Lower case capitalized words
        words that start with a capital letter, but not all capital words such as USA.
	2. padding 0s and tag is  <pad> 
        Identify the sentence with max length
        Append 0s at the end of shorter sentences to make them match this max length
        Set the tag for the 0s to <pad>
'''
def preprocess(df):
    def tolower(text):
        lowerlist=[]
        for word in text:
            pattern = re.compile('[A-Z][a-z]+')
            if re.match(pattern,word):
                cleantext1 = re.sub(pattern, word.lower(), word)
                lowerlist.append(cleantext1)
            else:
                lowerlist.append(word)
        return lowerlist
    def getMaxlength(words):
        sentlist = []
        sentence = []
        for word in words:
            if(word == '-DOCSTART-'):
                sentlist.append(sentence)
                sentence = []
                continue
            else:
                sentence.append(word) 
        maxlen = max(len(x) for x in sentlist ) 
        print('maximum length of sentence in the data is ', maxlen)
        print('sent with max length is -> ', max((x) for x in sentlist) )
        print('Number of sentences are ', len(sentlist))
        return maxlen, len(sentlist)
    
    df['word'] = tolower(df['word'])
    maxsentlen, NoOfSents = getMaxlength(df['word'])
    start = 0
    index = 0
    newwords = []
    newpostag = []
    newchunktag = []
    newnertag = []
    for index in range(0, len(df)):
        if(df.iloc[index]['word'] != '-DOCSTART-'):            
            newwords.append(df.iloc[index]['word'])
            newpostag.append(df.iloc[index]['pos'])
            newchunktag.append(df.iloc[index]['chunk'])
            newnertag.append(df.iloc[index]['ner'])
            #index += 1
            start += 1
        else:
            for i in range(start, maxsentlen):
                newwords.append(0)
                newpostag.append(0)
                newchunktag.append(0)
                newnertag.append('<pad>')            
            start = 0
    
    newdf = pd.DataFrame(list(zip(newwords, newpostag, newchunktag, newnertag)),  columns =['word', 'pos', 'chunk', 'ner']) 
            
    return newdf

'''
This method takes care of reading double quotes and NAs as a 
'''
def prepreprocessing(path, filename):
    fin = open(path + filename, "rt")
    fout = open(path + 'newdata/'+ filename, "w+")

    for line in fin:
        line = line.replace('"', '/"')
        fout.write(line.replace('NA', 'None'))
	
    fin.close()
    fout.close()

'''
def load_glove(word_index):
    EMBEDDING_FILE = 'C:\\Users\\Hari Ravella\\Downloads\\GoogleNews-vectors-negative300.bin'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class ToyNN(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers):
        super(self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        
    def forward(self, inp, hidden):
        return self.gru(self.embedding(inp), hidden)
    
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

'''

def main():
    path = 'D:/Spring 2020/assignments/RNN/NERCONLL2013/data/'

    prepreprocessing(path, "train.txt")
    traindata = pd.read_csv(path+ 'newdata/'+ 'train.txt', sep= ' ')
    #validata= pd.read_csv(path +"valid.txt", sep= ' ')
    #testdata= pd.read_csv(path +"test.txt", sep= ' ')

    print(traindata.head(10))
    print(traindata.columns)
    print(traindata.shape)
    traindata.columns = ['word', 'pos', 'chunk', 'ner']
    
    print(traindata.to_csv(path + 'new.csv'))
    newdf = preprocess(traindata)
    print('Ok dataframe ready')
    print(newdf.to_csv(path +'trainNew.txt'))
    #traindata.rename(columns=['word', 'pos', 'chunk', 'ner'], inplace = True)
    # text, POS tag, NE tag, I or O






if __name__ == "__main__":
    main()










































