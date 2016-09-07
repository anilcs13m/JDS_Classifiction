import os
import re
import nltk
import math
import pandas as pd
import numpy as np
from num2words import num2words
# from bs4 import BeautifulSoup
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
# we had removed all the rows which have NA value for jds     #
###############################################################

## read train data
train = pd.read_csv('train.csv')
#
print("just checking shape and columns names of training data")
# print(train.shape)
# print(train.columns.values)
#
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
#
# just viewing some of the text file
print('The firstjds is:')
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
        #description = yearexp(description) ## function call for handel years of exp
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
        #
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
            
        # Return a list of words
        return(words)


################################################################
# Initialize an empty list to hold the clean train description #
################################################################

clean_train_descriptions = []
for i in range( 0, len(train["jds"])):
	clean_train_descriptions.append(" ".join(description_to_wordlist(train["jds"][i], True)))
#
###############################################################
# Create an empty list and append the clean test description  #
###############################################################

clean_test_descriptions = []
for i in range(0,len(test["jds"])):
	clean_test_descriptions.append(" ".join(description_to_wordlist(test["jds"][i], True)))


print("**********Creating the bag of words of descriptions**************\n")
#
#####################################################################
# creation of bag of words can be done by using "CountVectorizer"   #
# Initialize the "CountVectorizer" object, which is scikit-learn's  #
# bag of words tool.                                                #
#####################################################################
#
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, 
                            preprocessor = None, stop_words = None, 
                            max_features = 2000)
######################################################################
#    fit_transform() does two functions:                             #
#    1. First, it fits the model and learns the vocabulary;          #
#    2. second, it transforms our training data into feature vectors.#
#    The input to fit_transform should be a list of strings.         # 
######################################################################
#
# Numpy arrays are easy to work with, so convert the result to an array
#
train_data_features = vectorizer.fit_transform(clean_train_descriptions)
#
train_data_features = train_data_features.toarray()
#
#####################################################################
# Get a bag of words for the test set, and convert to a numpy array #
#####################################################################
#
test_data_features = vectorizer.transform(clean_test_descriptions)
#
test_data_features = test_data_features.toarray()
#
forest = RandomForestClassifier(n_estimators = 100, n_jobs = -1, verbose = 1 )
#
forest = forest.fit( train_data_features, train["function"] )
#
# Use the random forest to make label predictions
#   
result = forest.predict(test_data_features)
#
for i, v in enumerate(result):
    print("function for jds :",i ,"-",v)
