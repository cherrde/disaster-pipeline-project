import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sqlalchemy import inspect

#load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
inspector = inspect(engine)
for table_name in inspector.get_table_names():
    print(table_name)
