
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
            print(word)
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
            if(word == '.'):
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
    for i in range(0, NoOfSents):
        if(df['word'][index] != '.'):            
            newwords.append(df['word'][index])
            newpostag.append(df['pos'][index])
            newchunktag.append(df['chunk'][index])
            newnertag.append(df['ner'][index])
            index += 1
            start += 1
        else:
            for i in range(start, maxsentlen):
                newwords.append(0)
                newpostag.append(0)
                newchunktag.append(0)
                newnertag.append('<pos>')            
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










































