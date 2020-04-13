
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
	lookup the representation for every word in the dataset's vocabulary
	------dimensions--------
	input: |V| x 300
	H: 256
	Output: 10


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

	References:
	https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

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
from gensim import models


import torch
import torch.nn as nn
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
        #print('sent with max length is -> ', max((x) for x in sentlist) )
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
def cleaning(path, filename):
    fin = open(path + filename, "rt")
    fout = open(path + 'newdata/'+ filename, "w+")

    for line in fin:
        line = line.replace('"', '/"')
        fout.write(line.replace('NA', 'None'))
	
    fin.close()
    fout.close()

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.i2o = nn.Linear(inhidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def RNN_train(model, train_X, train_y, val_X, val_y, lr, epochs=2, batch_size=2000):
    valid_preds = np.zeros((val_X.size(0)))
    trainloss = []
    testloss = []
    testaccuracy = []

    for e in range(epochs):
        start_time = time.time()

        optimizer = optim.SGD(model.parameters(), lr=lr)  #Optimizing with Stochastic Gradient Descent
        loss_fn = nn.MSELoss()  # Mean Squared Error Loss

        #Convert to tensor data
        train_tsdata = torch.utils.data.TensorDataset(train_X, train_y)
        valid_tsdata = torch.utils.data.TensorDataset(val_X, val_y)

        #Feed the tensor data to data loader. This partitions the data based on batch size
        train_loader = torch.utils.data.DataLoader(train_tsdata, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_tsdata, batch_size=batch_size, shuffle=False)

        #Run the model on train set and capture loss
        model.train()
        avg_loss = 0.
        for x_batch, y_batch in train_loader:
            X = Variable(torch.FloatTensor(x_batch))
            y = Variable(torch.FloatTensor(y_batch))
            optimizer.zero_grad() #null the gradients
            y_pred = model(X) #forward pass
            loss = loss_fn(y_pred.squeeze(), y.squeeze()) #Compute loss
            loss.backward() #back propagate
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        trainloss.append(avg_loss)

        #Run the model on validation set and capture loss
        model.eval()
        avg_val_loss = 0.
        testacc = 0
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            X_val = Variable(torch.FloatTensor(x_batch))
            y_pred_val = model(X_val)
            avg_val_loss += loss_fn(y_pred_val, y_batch.float()).item() / len(valid_loader)
            valid_preds[i * batch_size:(i+1) * batch_size] = y_pred_val[:, 0].data.numpy()
            testacc += np.sum(np.round(y_pred_val[:, 0].data.numpy()) == y_batch.float()[:, 0].data.numpy())
        elapsed_time = time.time() - start_time

        if (e % 100 == 99):
            print('\t Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(e + 1, epochs, avg_loss, avg_val_loss, elapsed_time))

        testloss.append(avg_val_loss)
        testaccuracy.append(testacc/ len(val_y))
    #Visualize the trainloss, validation loss and validation accuracy
    plt.title("plot of train,val loss and val accuracy for lr = {}".format(lr))
    plt.plot(trainloss)
    plt.plot(testloss)
    plt.plot(testaccuracy)
    plt.show()
    return min(testloss)

def main():

	cleaning("/Users/gkbytes/NER/conll2003/", "train.txt")
	cleaning("/Users/gkbytes/NER/conll2003/", "valid.txt")
	cleaning("/Users/gkbytes/NER/conll2003/", "test.txt")


	traindata= pd.read_csv("/Users/gkbytes/NER/conll2003/newdata/train.txt", sep= ' ')
	validata= pd.read_csv("/Users/gkbytes/NER/conll2003/newdata/valid.txt", sep= ' ')
	testdata= pd.read_csv("/Users/gkbytes/NER/conll2003/newdata/test.txt", sep= ' ')

	print(traindata.head(10))
	print(traindata.columns)

	print(traindata.shape)
	traindata.columns = ['word', 'pos', 'chunk', 'ner']

	newdf = preprocess(traindata)
	print('Ok dataframe ready')
	#traindata.rename(columns=['word', 'pos', 'chunk', 'ner'], inplace = True)
	# text, POS tag, NE tag, I or O

	
	w = models.KeyedVectors.load_word2vec_format('/Users/gkbytes/NER/conll2003/GoogleNews-vectors-negative300.bin', binary=True)

	print(w[1])
	print(type(w))

	n_input=300
	n_hidden = 256
	n_output= 10
	rnn = RNN(n_input,n_hidden,n_output)


 
	#input size-- |V| x 300
	# output size

if __name__ == "__main__":
    main()










































