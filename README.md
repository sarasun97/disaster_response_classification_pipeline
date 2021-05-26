# Disaster response classification pipeline project
Part of the Udacity Data Scientist Nanodegree Program

## Table of Contents
1. Installation
2. Project Motivation
3. File Descriptions
4. Instruction
5. Screenshot
6. Discussion
7. Licensing, Authors, and Acknowledgements

## 1. Installation
  [python (>=3.6)](https://www.python.org/downloads/)\
  [sys](https://docs.python.org/3/library/sys.html)\
  [pandas](https://pandas.pydata.org/)\
  [numpy](https://numpy.org/)\
  [sklearn](https://sklearn.org/)\
  [nltk](https://www.nltk.org/)\
  [html.parser](https://docs.python.org/3/library/html.parser.html)\
  [pattern3](https://pypi.org/project/pattern3/)\
  [sqlalchemy](https://www.sqlalchemy.org/)\
  [plotly](https://plotly.com/python/)\
  [joblib](https://joblib.readthedocs.io/en/latest/)\
  [flask](https://flask.palletsprojects.com/en/2.0.x/)

## 2. Project Motivation
This project is part of the Udacity Data Scientist Nanodegree Program. 

After a natural disaster, thousands of people will send out messages to ask for help through various channels such as social media. For example, I need food; I am trapped under the rubble. However, the government does not have enough time to read all the messages and send them to various departments. Then, this project will play an important role. 

The project use data from [Figure Eight](https://appen.com/) (which has been acquired by Appen) to build a model for an API that classifies disaster messages. The project includes a web app where an emergency worker can input a new message and get classification results in several categories.

The dataset contains more than 26248 messages drawn from events including the 2010 Haiti earthquake, the Chile earthquake of 2010, Pakistan Floods of 2010. The messages have been classified into 36 categories after being translated to English from the original language.

## 3. File Descriptions
├── _data\
│ ├── DisasterResponse.db>>> \
│ ├── disaster_categories.csv>>> \
│ ├── disaster_messages.csv\
│ └── process_data.py >>> ETL pipeline - a Python script that loads the messages and categories datasets\
                          merges the two datasets,cleans the data,stores it in a SQLite database\
├── _models\
│ ├── Text_Length_Extractor.py\
│ ├── Text_Normalization_Function.py\
│ ├── classifier.pkl\
│ └── train_classifier.py >>> ML pipeline - a Python script that builds a text processing and machine learning pipeline\
                              which trains and tunes a model using GridSearchCV, and then exports the final model as classifier.pkl\
├── _web_app\
│ ├── _templates\
│ │ ├─ go.html\
│ │ ├─ distribution.png\
│ │ └─ master.html\

│ ├── Text_Length_Extractor.py\
│ ├── Text_Normalization_Function.py\
│ └── run.py

## 4. Instruction
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the web_app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## 5. Screenshot
![the home page of the web app](https://github.com/sarasun97/Disaster-response-classification-pipeline-project/tree/main/screenshot/screenshot1.png)\
![classification example 1](https://github.com/sarasun97/Disaster-response-classification-pipeline-project/tree/main/screenshot/screenshot2.png)\
![classification example 2](https://github.com/sarasun97/Disaster-response-classification-pipeline-project/tree/main/screenshot/screenshot3.png)

## 6. Discussion
This dataset is imbalanced. Some labels like tools have few examples, child_alone category had 0 instances. The imbalance will definitely affect the performance of the model, especially recall(TP/(TP+FN)) and precision(TP(TP+FP). 

## 7. Licensing, Authors, and Acknowledgments
Authors:
Sara Sun

Acknowledgements
I have learned a lot from [Data Scientist Nanodegree Program](https://classroom.udacity.com)

The data is provided by [Figure Eight](https://appen.com/)

I learned a lot from [Maria Vaghani's project](https://github.com/mariavaghani/Disaster-Response-messages-NLP) and [Evans Doe Ocansey's project] (https://github.com/evansdoe/disaster-response-pipeline)
