
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



def main():

    traindata= pd.read_csv("/Users/gkbytes/NER/conll2003/train.txt", sep= '\t')
    validata= pd.read_csv("/Users/gkbytes/NER/conll2003/valid.txt", sep= '\t')
    testdata= pd.read_csv("/Users/gkbytes/NER/conll2003/test.txt", sep= '\t')

    print(traindata.head(10))
    print(traindata.columns)
    # text, POS tag, NE tag, I or O






if __name__ == "__main__":
    main()










































