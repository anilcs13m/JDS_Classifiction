import os
import re
import nltk
import math
import logging
import nltk.data
import pandas as pd
import numpy as np
from num2words import num2words
# from bs4 import BeautifulSoup
from gensim.models import word2vec
from sklearn.preprocessing import Imputer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

###############################################################
# The file contains function job_title and job description    #
# over task is to develop a model that predict the function   #
# for a given job description                                 #
###############################################################

###############################################################
# Import the pandas package, then use the "read_csv" function #
# to read the labeled training data                           #
# train file is created after pre-processing of given problem  #
# we had removed all the data which have NA value for jds     #
###############################################################

## input train file
train = pd.read_csv('train.csv')
###############################

print("just checking shape and columns names of training data")
# print(train.shape)
# print(train.columns.values)

"""
***********************
        change this file according to your test file,
        this file i just created for testing
********************** 
"""
## input your test file
test = pd.read_csv('test.csv')
##############################

print("just checking shape and columns names of test data")
# print(test.shape)
# print(test.columns.values)

# just viewing some of the text file
print('The first jds is:')
# print(train["jds"][0])

def yearexp(description):
    #
    # This function return years of exp in words like 
    # 1 year exp return as one year
    #
    words = description.split()
    target = "years"
    #
    # this list is use to conver years of exp upto 25 into word
    #
    lis = ['1','2','3','4','5','6','7',
            '8','9','10','11','12','13',
            '14','15','16','17','18','19',
            '20','21','22','23','24','25']
    for i,w in enumerate(words):
        if w == target and words[i-1] in lis:
            word = num2words(int(words[i-1]))
            description = description.replace(words[i-1],word)
        if w == target and words[i-2] in lis:
            word1 = num2words(int(words[i-2])) 
            description = description.replace(words[i-2],word1)
    return description

# some preprocessing and text data cleaning
def description_to_wordlist(description, remove_stopwords=False ):
        #
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        description = description.lower()
        # description = BeautifulSoup(description).get_text()
        description = description.replace(" oo "," object oriented ")
        description = description.replace("oops", " object oriented programming ")
        description = description.replace("c++","cpp")
        description = description.replace("c#","csharp")
        description = description.replace("unity3d"," unitythreed ")
        description = description.replace(" fb "," facebook ")
        description = description.replace(" g+ "," googleplus ")
        description = description.replace(" pro ", " professional ")
        description = description.replace(" 2d "," twod ")
        description = description.replace(" 3d "," threed ")
        description = description.replace("yrs", " years ")
        # description = yearexp(description) ## function call for handel years of exp
        description = description.replace("10th","tenth")
        description = description.replace("12th"," twelfth")
        description = description.replace("r&d"," research and development ")
        description = description.replace("m.tech"," master of technology ")
        description = description.replace("b.tech"," bachelor of technology ")
        description = description.replace("b.e"," bachelor of engineering ")
        description = description.replace("m.e"," master of engineering ")
        description = description.replace(" ms "," master of science ")
        description = description.replace("m.s"," master of science ")
        description = description.replace("b.s"," bachelor of science ")
        description = description.replace(" pg "," postgraduate ")
        description = description.replace(" post graduate "," postgraduate ")
        description = description.replace("e-commerce"," ecommerce ")
        description = description.replace(" re-"," re")
        description = description.replace(" ca "," chartered accountant ")
        description = description.replace(" year "," years ")
        description = description.replace(" min "," minimum ")
        description = description.replace(" max "," maximum ")
        description = description.replace(" exp "," experience ")
        description = description.replace("sr."," senior ")
        description = description.replace("〈br〉", " ")
        description = description.replace("〈 /span〉"," ")
        description = description.replace(" to "," ")
        description = description.replace("%"," percentage ")
        description = description.replace("etc."," ")
        description = description.replace(" o "," ")
        description = description.replace("co-"," co")
        description = re.sub("[^a-zA-Z0-9]"," ",description)
        description = yearexp(description) ## function call for handel years of exp
        description = re.sub("[^a-zA-Z]"," ",description)
        #
        # split 
        words = description.split()
        #
        # Optionally remove stop words
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]

        # Return a list of words
        return(words)

################################################################
# Initialize an empty list to hold the clean train description #
################################################################
#
# clean_train_descriptions = []
#
# for i in range( 0, len(train["jds"])):
	# clean_train_descriptions.append(" ".join(description_to_wordlist(train["jds"][i], True)))
#
###############################################################
# Create an empty list and append the clean test description  #
###############################################################
#
# clean_test_descriptions = []
#
# for i in range(0,len(test["jds"])):
	# clean_test_descriptions.append(" ".join(description_to_wordlist(test["jds"][i], True)))

def descriptions_to_sentences( description, tokenizer, remove_stopwords=False ):
        #
        #  Function to split a description into parsed sentences. Returns a
        #  list of sentences, where each sentence is a list of words
        #
        # Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(description.strip())
        #
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call description_to_wordlist to get a list of words
                sentences.append(description_to_wordlist(raw_sentence, remove_stopwords ))
        # return sentance
        return sentences

def getCleanDescription(descriptions):
    #
    clean_descriptions = []
    #
    for description in descriptions["jds"]:
        #
        clean_descriptions.append(description_to_wordlist(description, remove_stopwords=True ))
        #
    return clean_descriptions

def makeFeatureVec(words, model, num_features):
    #
    # Function to average all of the word vectors in a given
    # paragraph
    # Pre-initialize an empty numpy array (for speed)
    #
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    #
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the description and, if it is in the model's
    # vocaublary, add its feature vector to the total
    #
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    #
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(descriptions, model, num_features):
    #
    # Given a set of description (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    # Initialize a counter
    #
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    #
    descriptionFeatureVecs = np.zeros((len(descriptions),num_features),dtype="float32")
    #
    # Loop through the descriptions
    #
    for description in descriptions:
       #
       # Print a status message every 1000th descriptions
       #
       if counter%1000. == 0.:
           print("Description %d of %d" % (counter, len(descriptions)))
       #
       # Call the function (defined above) that makes average feature vectors
       #
       descriptionFeatureVecs[counter] = makeFeatureVec(description, model, num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return descriptionFeatureVecs


# Load the punkt tokenizer
#
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#
#
# Split the labeled and unlabeled training sets into clean sentences
sentences = []  # Initialize an empty list of sentences
for desc in train["jds"]:
    sentences += descriptions_to_sentences(desc, tokenizer)


# Set values for various parameters
num_features = 400    # Word vector dimensionality
min_word_count = 50   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words
#
# Initialize and train the model (this will take some time)
#
print("Training Word2Vec model")
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, 
                            min_count = min_word_count,  
                            window = context, 
                            sample = downsampling, seed=1)

###############################################################
#
# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
#
model.init_sims(replace=True)
#
# Create average vectors for the training and test sets
#
trainDataVecs = getAvgFeatureVecs( getCleanDescription(train), model, num_features )
#
testDataVecs = getAvgFeatureVecs(getCleanDescription(test), model, num_features )
#
trainDataVecs = Imputer().fit_transform(trainDataVecs)
#
forest = RandomForestClassifier(n_estimators =100)
# forest = RandomForestClassifier()
#
forest = forest.fit(trainDataVecs, train["function"] )
#
# Use the random forest to make label predictions
#     print "Predicting test labels...\n"
result = forest.predict(testDataVecs)
#
for i, v in enumerate(result):
    print("function for jds :",i ,"-",v)
# f