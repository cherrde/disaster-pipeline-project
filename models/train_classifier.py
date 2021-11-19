# import libraries

import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Loads data into a number of datafreames

    Args:
        (str) database_filepath - filepath to datab ase to craete a dataframe
    Returns:
        (dataframe) dfx - a dataframe with messages data
        (dataframe) dfy - a dataframe with Y data
        (list) listc - list of column names
    """
    sqlite_prefix = 'sqlite:///'
    sqlite_full = sqlite_prefix + database_filepath
    engine = create_engine(sqlite_full)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    dfx = df['message']
    dfy = df.iloc[:, 4:]
    listc = dfy.columns.values

    return dfx, dfy, listc

def tokenize(text):
    """
    Tokenizes text strings

    Args:
        (str) text - A string to tokenize
    Returns:
        (list) clean_tokens - list of cleaned tokens
    """
    text = re.sub(r"[^a-zA-Z]", " ", str(text))

    words = word_tokenize(text)
    
    words = [w for w in words if w not in stopwords.words("english")]
    
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for words in words:
        clean_tok = lemmatizer.lemmatize(words).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a model using a pipeline

    Args:
        none
    Returns:
        (model) cv - a model to train
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    # ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    # ('clf', MultiOutputClassifier(DecisionTreeClassifier()))])
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))])
    parameters = {
        'clf__estimator__n_estimators': [8] #,
        # 'clf__estimator__min_samples_split': [2, 3, 4],
        # 'clf__estimator__max_depth': [4, 6, 8]
        # 'clf__estimator__max_depth': [5] # use only with DecisionTreeClassifier
         }
    
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=2, verbose=3)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate a model and print a classification report

    Args:
        model - a model to evaluate
        X_test - x test data
        Y_test - y test data
        category_names - list of categories
    Returns:
        (model) cv - a model to train
    """
    y_pred_test = model.predict(X_test)
    print(classification_report(Y_test, y_pred_test, target_names=category_names))

def save_model(model, model_filepath):
    """
    Save a model as a pickle file

    Args:
        model - a model to save
        model_filepath - a filename and path for the pickle file
    Returns:
        none
    """
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()