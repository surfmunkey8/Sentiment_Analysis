from string import punctuation

import pandas as pd
import numpy as np
import re
import nltk
nltk.download()
import spacy
import string

from spacy.symbols import number
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

#Preprocessing the .csv files

'''
    Step 1:  Import the csv file into a pandas data frame.
    Step 2:  Remove the text column to be preprocessed.
    Step 3:  Converting the text field type to string to ensure consistency through out the entire data frame. 
    Step 4:  Convert all letters to lower case.
    Step 5:  Remove allow punctuation except '!'
    Step 6:  Check to see which reviews have '!' and store it in the dataframe.
    Step 7:  Check to see which reviews have "no" and store it in the dataframe.
    Step 8:  Counting all the words in the review
    Step 9:  Taking the natural log of each word count
    Step 10:  Remove all stop words
    
'''
#Importing the .csv file
origin_df = pd.read_csv('test_amazon.csv')

#Isolating the text column to its own data frame and adding columns for other features
reviews_df = pd.DataFrame()

reviews_df.loc[:, 'Text'] = origin_df.loc[:, "text"]


#Converting to string
reviews_df["Text"] = reviews_df["Text"].astype(str)


#Converting all letters to lowercase.
reviews_df["Text"] = reviews_df["Text"].str.lower()


#Removing all numbers from string
numbs = string.digits

def remove_numbers(text):
    return text.translate(str.maketrans('', '', numbs))

reviews_df["Text"] = reviews_df["Text"].apply(lambda digit: remove_numbers(digit))

#Removing all punctuation except !
punc = string.punctuation

punc = punc.replace('!', '')

def remove_punc(text):
    return text.translate(str.maketrans('', '', punc))

reviews_df["Text"] = reviews_df["Text"].apply(lambda text: remove_punc(text))


#Checking for which cells contain an exclamation and adding it to the dataframe.
reviews_df["Contains Exclamation"] = reviews_df["Text"].str.contains('!').astype(int)

'''
    Removing the exclamations after they had been counted
'''
def remove_exclamation(text):
    return text.translate(str.maketrans('', '', '!'))

reviews_df["Text"] = reviews_df["Text"].apply(lambda text: remove_exclamation(text))


#Checking for which cells contain "no" and adding it to the dataframe.
reviews_df["Contains No"] = reviews_df["Text"].str.contains('no').astype(int)


#Counting each word in a cell and adding it to the dataframe.
def word_count(text):
    return len(str(text).split())

reviews_df["Numb. of Words"] = reviews_df["Text"].apply(lambda text: word_count(text))


#Taking the natural log of each Word Count cell.
reviews_df["Numb. of Words"] = reviews_df["Numb. of Words"].astype(float)

reviews_df["Numb. of Words"] = reviews_df["Numb. of Words"].apply(lambda number: np.log(number))


#Removing stop words
stop_words = set(stopwords.words('english'))

def delete_stopwords(text):

    useful_words = word_tokenize(text)

    return [word for word in useful_words if word not in stop_words]

reviews_df["Text"] = reviews_df["Text"].apply(lambda text: delete_stopwords(text))

print(reviews_df.head())


#Count the positive Words
