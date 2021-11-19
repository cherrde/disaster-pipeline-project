# Disaster Response Pipeline Project
This project will perform three different functions. The first is to take an
amount of data in csv format including messages and categories of messages
gathered after a disaster and clean and format those bits of data then upload to
a SQLite database. One additional function is to create a table of most common
tokens for presenting a graphic in the third feature.  The second function is
to take the information from the database and generate a trained classifier
model based on the messages then save that model off to a file. The final
function is to run a web app presenting a pair of graphics and when entered,
classify a message in on of the 36 categories defined based on the model
previously generated.

## Motivation:
The reason behind this project was to support the effort discussed by Figure Eight
to potentially provide emergency aid workers a tool to classify messages
received via twitter or other mediums into buckets that will allow messages to
be routed to an organization that might be able to provide the necessary
assistance.

### File list:
  |--app
      |--templates
        |--go.html - an html template file
        |--master.html - an html template file
      |--run.py - a python script that runs a flask website using the templates above
  |--data
    |--DisasterResponse.db - A SQLite database file
    |--disaster_categories.csv - a csv file containing disaster message categories
    |--disaster_messages.csv - a csv file containing disaster messages   
    |--process_data.py - a python script that processes and cleans data
  |--models
    |--classifier.pkl - a pickle file containing a classifier model
    |--classifier_3params.pkl - a pickle file containing a model with 3 parameters for testing purposes
    |--train_classifier.py - a python script that creates and validates a models
  |--ETL Pipeline Preparation.ipynb - a Jupyter notebook used for testing
  |--ML Pipeline Preparation.ipynb - a Jupyter notebook used for testing
  |--README.md - this File

## Installation:

Download from GitHub. Run the scripts in order per the Instructions below. Ensure your
python environment has the following necessary packages:

numpy
pandas
nltk
sqlalchemy
collections
sys
re
plotly
json
flask
sklearn
pickle

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Acknowledgments:
Thanks to Udacity for providing templates and initial coding examples. Thanks the
Udacity Mentor program for assistance in troubleshooting code snippets and providing
support through out the process. Sci-Kit Learn documentation and pandas documentation
was used extensively to provide coding parameters. Finally thanks to Figure
Eight for providing a challenge and the initial data set used to work from.
