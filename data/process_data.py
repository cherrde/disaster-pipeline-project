# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import re
from collections import Counter

import nltk
nltk.download(['stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def load_data(messages_filepath, categories_filepath):
    """
    Loads data into a dataframe

    Args:
        (str) messages_filepath - filepath to csv file to craete a dataframe
        (str) categories_filepath - filepath to csv file to create a dataframe
    Returns:
        (dataframe) df - a dataframe with clean data
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, left_on = 'id', right_on='id')

    return df

def clean_data(df):
    """
    Loads data for the specified directory

    Args:
        (dataframe) df - a dataframe that needs data cleansed
    Returns:
        (dataframe) df - a dataframe with clean data
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:len(x) -2])
    # print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner', sort=False)

    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)
    
    # fix '2' values in related
    df.related.replace(2,1,inplace=True)

    return df

def save_data(df, database_filename):
    """
    saves data for the specified directory

    Args:
        (dataframe) df - name the dataframe to save to a database
        (str) database_filename - name for the database file
    Returns:
        none
    """
    sqlite_prefix = 'sqlite:///'
    sqlite_full = sqlite_prefix + database_filename
    engine = create_engine(sqlite_full)
    df.to_sql('Messages', engine, index=False, if_exists='replace')
    
def tokenize(text):
    """
    tokenizes, lemmetizes, stems and removes stopwords from text

    Args:
        (str) text - text to be tokenized
    Returns:
        (list) clean_tokens - a list of clean tokens
    """    
    text = re.sub(r"[^a-zA-Z]", " ", str(text))

    words = word_tokenize(text)
    
    words = [w for w in words if w not in stopwords.words('english')]
    
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for words in words:
        clean_tok = lemmatizer.lemmatize(words).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens    

def create_word_freq_table(database_filename):
    """
    tokenizes words and saves data for the specified directory

    Args:
        (str) database_filename - name for the database file
    Returns:
        none
    """
    sqlite_prefix = 'sqlite:///'
    sqlite_full = sqlite_prefix + database_filename
    engine = create_engine(sqlite_full)
    df = pd.read_sql("SELECT * FROM Messages", engine)      
    txt = df['message'].apply(tokenize)
    p = Counter(" ".join(list(map(str, txt))).split()).most_common(50)
    result = pd.DataFrame(p, columns=['Word', 'Frequency'])
    result.to_sql('MsgTokens', engine, index=False, if_exists='replace') 
    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')
        
        print('Create word frequency table for graphic use later...')
        create_word_freq_table(database_filepath)
        
        print('process data script complete!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
