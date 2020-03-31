
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