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

def tokenize(text):
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
  
def main():

    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql("SELECT * FROM Messages", engine)      
    txt = df['message'].apply(tokenize)
    p = Counter(" ".join(list(map(str, txt))).split()).most_common(50)
    result = pd.DataFrame(p, columns=['Word', 'Frequency'])
    result.to_sql('MsgTokens', engine, index=False, if_exists='replace')   

if __name__ == '__main__':
    main()
