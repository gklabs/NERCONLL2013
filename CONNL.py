
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

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

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


class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings):
        super(LSTMClassifier, self).__init__()
        
        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
        
        """
        
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        
    def forward(self, input_sentence, batch_size=None):
    
        """ 
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
        
        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)
        
        """
        
        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        #input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda()) # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        final_output = self.softmax(self.label(final_hidden_state[-1])) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        
        return final_output


def LSTM_train(model, train_X, train_y, val_X, val_y, lr, epochs=2, batch_size=2000):
    valid_preds = np.zeros((val_X.size(0)))
    trainloss = []
    testloss = []
    testaccuracy = []

    for e in range(epochs):
        start_time = time.time()

        optimizer = optim.Adam(model.parameters(), lr=lr)  #Optimizing with Stochastic Gradient Descent
        loss_fn = F.cross_entropy
        
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
    
    path = 'D:/Spring 2020/assignments/RNN/NERCONLL2013/data/'

    cleaning(path, "train.txt")
    cleaning(path, "valid.txt")
    cleaning(path, "test.txt")

    traindata= pd.read_csv(path+"/newdata/train.txt", sep= ' ')
    validata= pd.read_csv(path +"/newdata/valid.txt", sep= ' ')
    testdata= pd.read_csv(path +"/newdata/test.txt", sep= ' ')

    print(traindata.head(10))
    print(traindata.columns)

    print(traindata.shape)
    traindata.columns = ['word', 'pos', 'chunk', 'ner']

    newdf = preprocess(traindata)
    newdf_val = preprocess(validata)
    print('Ok dataframe ready')
    #traindata.rename(columns=['word', 'pos', 'chunk', 'ner'], inplace = True)
    # text, POS tag, NE tag, I or O

    
    google = models.KeyedVectors.load_word2vec_format('C:/Users/Hari Ravella/Downloads/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)

    #creating vocabulary
    def createvocab(text):
        V=[]
        for word in text:
            if word in V:
                continue
            else :
                V.append(word)
        return V
    
    vocab = createvocab(newdf.word)
    print(len(vocab))
    
    indexvecs = {wx: vocab.index(wx) for wx in vocab}
     
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found = 0
    
    #Create dictionary of words and representation from google news vector
    for i, word in enumerate(vocab):
        try: 
            word = str(word)
            weights_matrix[i] = google[word]
            words_found += 1
        except KeyError:
            print(word)
            weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))
    
    print(words_found)  
    
   
    
    
    #n_input=300
    #n_hidden = 256
    #n_output= 10
    wm = torch.from_numpy(weights_matrix).float()
    model = LSTMClassifier(batch_size= 2000, output_size= len(newdf.ner.unique()), hidden_size = 256, vocab_size = len(vocab), embedding_length = 300, word_embeddings= wm)
    
    train_X = torch.from_numpy(train_data.values).float()
    train_y = torch.from_numpy(train_labels.values.reshape((train_labels.shape[0],-1))).float()
    
    LSTM_train(model, train_X, train_y, val_X, val_y, lr, epochs=2, batch_size=2000)
    
    
    
    
    rnn = RNN(n_input,n_hidden,n_output)


 
    #input size-- |V| x 300
    # output size

if __name__ == "__main__":
    main()










































