# JDS_Classifiction

For A give Job discription, task is to predict the function for given JDS

 * python3
 * num2words
 * word2vec
 * pandas
 * numpy
 * nltk

nltk with stopwords corpus


for testing just change test.csv file with your test file with same name as test.csv in this directory

or you can change in program 
## input your test file
test = pd.read_csv('test.csv')

There are two model

"rf_model.py" model using random forest
"rf_word2vec.py" model using random forest + word2vec

Run keeping all the file in the same directory

Terminal

python3 rf_model.py

python3 rf_word2vec.py
it will take some time
 
output as a function

