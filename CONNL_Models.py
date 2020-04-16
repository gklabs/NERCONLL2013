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


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, word_embeddings):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class GRUNet(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings, bidirectional):
        super(GRUNet, self).__init__()
        
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        
        self.bidirectional = bidirectional
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.gru = nn.GRU(embedding_length, hidden_size, bidirectional=bidirectional)
        self.label = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        
    def init_hidden(self,  biflag, batch_size):
        return Variable(torch.zeros(biflag, batch_size, self.hidden_size))
               
    def forward(self, input_sentence, batch_size=None):
        input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
        if(self.bidirectional == True):
            self.hidden = self.init_hidden(2, batch_size)
        else:
            self.hidden = self.init_hidden(1, batch_size)
        output, h = self.gru(input, self.hidden)
        final_out = self.softmax(self.label(output[-1]))
        return final_out
    
    

class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings, bidirectional):
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
        
        self.bidirectional = bidirectional
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(embedding_length, hidden_size, bidirectional=bidirectional)
        self.label = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
        
    def init_hidden(self, biflag, batch_size):
        return(Variable(torch.randn(biflag, batch_size, self.hidden_size)),
                        Variable(torch.randn(biflag, batch_size, self.hidden_size)))

    def forward(self, input_sentence, batch_size=None):
        #print('has it come here')
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
        input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
        if(self.bidirectional == True):
            self.hidden = self.init_hidden(2, batch_size)
        else:
            self.hidden = self.init_hidden(1, batch_size)
        
        '''
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)) # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)) # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        '''
        output, (final_hidden_state, final_cell_state) = self.lstm(input, self.hidden)
        final_output = self.softmax(self.label(final_hidden_state[-1])) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        
        return final_output

class RNN(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings, bidirectional):
        super(RNN, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.bidirectional = bidirectional
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.rnn = nn.RNN(embedding_length, hidden_size, num_layers=1, bidirectional=bidirectional)
        if(self.bidirectional == True):
            self.label = nn.Linear(2*hidden_size, output_size)
        else:
            self.label = nn.Linear(hidden_size, output_size)
    
    def init_hidden(self, biflag, batch_size):
        return Variable(torch.zeros(biflag, batch_size, self.hidden_size))
        #return(Variable(torch.randn(biflag, batch_size, self.hidden_size)), Variable(torch.randn(biflag, batch_size, self.hidden_size)))
    
    def forward(self, input_sentences, batch_size=None):
        
        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        
        if(self.bidirectional == True):
            self.hidden = self.init_hidden(2, batch_size)
        else:
            self.hidden = self.init_hidden(1, batch_size)
        '''
        if batch_size is None:
            h_0 = Variable(torch.zeros(4, self.batch_size, self.hidden_size).cuda()) # 4 = num_layers*num_directions
        else:
            h_0 =  Variable(torch.zeros(4, batch_size, self.hidden_size).cuda())
        '''
        output, h_n = self.rnn(input, self.hidden)
        # h_n.size() = (4, batch_size, hidden_size)
        h_n = h_n.permute(1, 0, 2) # h_n.size() = (batch_size, 4, hidden_size)
        h_n = h_n.contiguous().view(h_n.size()[0], h_n.size()[1]*h_n.size()[2])
        # h_n.size() = (batch_size, 4*hidden_size)
        logits = self.label(h_n) # logits.size() = (batch_size, output_size)
        
        return logits
    
def Training(model, train_X, train_y, val_X, val_y, lr, epochs=2, batch_size=2000):
    valid_preds = np.zeros((val_X.size(0)))
    trainloss = []
    testloss = []
    testaccuracy = []
    print('start training')
    for e in range(epochs):
        start_time = time.time()

        optimizer = optim.Adam(model.parameters(), lr=lr)  #Optimizing with Stochastic Gradient Descent
        loss_fn = F.cross_entropy
        
        #Convert to tensor data
        train_tsdata = torch.utils.data.TensorDataset(train_X, train_y)
        valid_tsdata = torch.utils.data.TensorDataset(val_X, val_y)
        print('Converted to tensor data')
        #Feed the tensor data to data loader. This partitions the data based on batch size
        train_loader = torch.utils.data.DataLoader(train_tsdata, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_tsdata, batch_size=batch_size, shuffle=False)
        print('Mini batching done')
        if(train_loader == None):
            print('null object')
        #Run the model on train set and capture loss
        else:
            model.train()
            avg_loss = 0.
            for x_batch, y_batch in train_loader:
                X = Variable(torch.LongTensor(x_batch))
                y = Variable(torch.LongTensor(y_batch.long()))
                #print('created X and y ', X.shape, y.shape)
                optimizer.zero_grad() #null the gradients
                #print('null gradients')
                y_pred = model(X, X.shape[0]) #forward pass
                #print('Called model.forward')
                #y_batch = torch.FloatTensor(y_batch)
                #print(y_batch.dtype)
                #print(y.dtype)
                loss = loss_fn(y_pred.squeeze(), y.squeeze()) #Compute loss
                #print('Loss = ')
                #print(loss)
                loss.backward() #back propagate
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
                #print('avg_loss per batch = ')
                #print(avg_loss)
            trainloss.append(avg_loss)
            print('loss training = ', avg_loss)
            #Run the model on validation set and capture loss
            model.eval()
            avg_val_loss = 0.
            testacc = 0
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                X_val = Variable(torch.LongTensor(x_batch))
                y_val = Variable(torch.LongTensor(y_batch.long()))
                y_pred_val = model(X_val, X_val.shape[0])
                avg_val_loss += loss_fn(y_pred_val.squeeze(), y_val.squeeze()).item() / len(valid_loader)
                valid_preds[i * batch_size:(i+1) * batch_size] = y_pred_val[:, 0].data.numpy()
                testacc += np.sum(np.round(y_pred_val[:, 0].data.numpy()) == y_batch.float()[:, 0].data.numpy())
            elapsed_time = time.time() - start_time
    
            #if (e % 100 == 99):
            print('\t Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(e + 1, epochs, avg_loss, avg_val_loss, elapsed_time))
            print(np.round(y_pred_val[:, 0].data.numpy()))
            testloss.append(avg_val_loss)
            testaccuracy.append(testacc/ len(val_y))
    print(testaccuracy)
    
    #Visualize the trainloss, validation loss and validation accuracy
    plt.title("plot of train,val loss and val accuracy for lr = {}".format(lr))
    plt.plot(trainloss)
    plt.plot(testloss)
    plt.plot(testaccuracy)
    plt.show()
    return min(testloss), valid_preds


##############################################################################################
#####################MAIN






path = 'D:\\Spring 2020\\assignments\\RNN\\NERCONLL2013\\data\\newdata\\'
traindata= pd.read_csv(path + "train.txt", sep= ' ')
validata= pd.read_csv(path + "valid.txt", sep= ' ')
testdata= pd.read_csv(path + "test.txt", sep= ' ')

print(traindata.head(10))
print(traindata.columns)

print(traindata.shape)
traindata.columns = ['word', 'pos', 'chunk', 'ner']
validata.columns = ['word', 'pos', 'chunk', 'ner']

newdf = preprocess(traindata)
print('traindone')
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
        weights_matrix[i] = np.zeros((300,))#np.random.normal(scale=0.6, size=(300, ))

print(words_found)  
          
#Gather the indixes for words from vocabulary
indexvecs = {wx: vocab.index(wx) for wx in vocab}

#Get words from train data as X
train_X = newdf.word.copy()
#SUbset because this lookup is taking time
words = train_X[train_X != 0]
indices = words.index
#Lookup the indices of the words in training data row by row from indexvecs created above
for i, word in zip(indices, words):
  train_X[i] = indexvecs[word] if word in indexvecs else 0 #] for word in sentence]
#For zeros assign it seperately
zeroindex = newdf[newdf.word == 0].index
train_X[zeroindex] = indexvecs['0']

#Perform above steps for validation
val_X = newdf_val.word.copy()
words_v = val_X[val_X != 0]
indices_v = words_v.index
#Lookup
for i, word in zip(indices_v, words_v):
  val_X[i] = indexvecs[word] if word in indexvecs else 0 #] for word in sentence]
#Same steps for assign zero indices
zeroindex_v = newdf_val[newdf_val.word == 0].index
val_X[zeroindex_v] = indexvecs['0']


print(len(train_X))
print(len(newdf))
print(len(val_X))
print(len(newdf_val))




############################################################################################################
######################          MODELLING


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
le = preprocessing.LabelEncoder()
ytrain = le.fit_transform(newdf.ner.values)
ytest = le.transform(newdf_val.ner.values)
train_Xf= np.array(train_X.values.astype(int).reshape((newdf.shape[0],-1)))
trainX = torch.from_numpy(train_Xf).long().to(device)
trainy = torch.from_numpy(ytrain.reshape((newdf.shape[0],-1))).long().to(device)

val_Xf= np.array(val_X.values.astype(int).reshape((newdf_val.shape[0],-1)))
valX = torch.from_numpy(val_Xf).long().to(device)
valy = torch.from_numpy(ytest.reshape((newdf_val.shape[0],-1))).long().to(device)
print(trainX.shape)
print(trainy.shape)
print(valX.shape)
print(valy.shape)

wm = torch.from_numpy(weights_matrix).float().to(device)

#LSTM
model_lstm = LSTMClassifier(batch_size= 2000, output_size= len(newdf.ner.unique()), hidden_size = 256, vocab_size = len(vocab), embedding_length = 300, word_embeddings= wm, bidirectional = False)
l_l, k_l = Training(model_lstm, trainX, trainy, valX, valy, lr= 0.0001, epochs=2, batch_size=2000)

model_bi_lstm = LSTMClassifier(batch_size= 2000, output_size= len(newdf.ner.unique()), hidden_size = 256, vocab_size = len(vocab), embedding_length = 300, word_embeddings= wm, bidirectional = True)
l_bi_l, k_bi_l = Training(model_bi_lstm, trainX, trainy, valX, valy, lr= 0.0001, epochs=2, batch_size=2000)

#RNN
model_rnn = RNN(batch_size= 2000, output_size= len(newdf.ner.unique()), hidden_size = 256, vocab_size = len(vocab), embedding_length = 300, word_embeddings= wm, bidirectional = False)
l_r, k_r = Training(model_rnn, trainX, trainy, valX, valy, lr= 0.0001, epochs=2, batch_size=2000)

model_bi_rnn = RNN(batch_size= 2000, output_size= len(newdf.ner.unique()), hidden_size = 256, vocab_size = len(vocab), embedding_length = 300, word_embeddings= wm, bidirectional = True)
l_bi_r, k_bi_r = Training(model_bi_rnn, trainX, trainy, valX, valy, lr= 0.0001, epochs=2, batch_size=2000)

#GRU
model_gru = GRUNet(batch_size= 2000, output_size= len(newdf.ner.unique()), hidden_size = 256, vocab_size = len(vocab), embedding_length = 300, word_embeddings= wm, bidirectional = False)
l_g, k_g = Training(model_gru, trainX, trainy, valX, valy, lr= 0.0001, epochs=2, batch_size=2000)

model_bi_gru = GRUNet(batch_size= 2000, output_size= len(newdf.ner.unique()), hidden_size = 256, vocab_size = len(vocab), embedding_length = 300, word_embeddings= wm, bidirectional = True)
l_bi_g, k_bi_g = Training(model_bi_gru, trainX, trainy, valX, valy, lr= 0.0001, epochs=2, batch_size=2000)