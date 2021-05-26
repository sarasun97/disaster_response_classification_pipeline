# import libraries
import sys
from sqlalchemy import create_engine
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report,confusion_matrix, precision_score,\
recall_score,accuracy_score,  f1_score,  make_scorer
from Text_Normalization_Function import normalize_corpus
from Text_Length_Extractor import Text_Length_Extractor

import pickle

def load_data(database_filepath):
    """
    INPUT:
        database_filepath - Python str object - path to the database that stored data
        
    OUTPUT:
        X - A pd series holds messages
        y - A pd dataframe holds all the categories for these messages.
        category_names- A list holds all the category names for these messages.
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM disaster_data", engine)
    X = df.message
    y = df.loc[:,"related":"direct_report"]
    category_names=y.columns
    return X, y,category_names

def build_model():
    """
    INPUT:
        None
        
    OUTPUT:
        GridSearchCV_pipeline - A grid-search pipeline used to train the model\
        and find the best parameters
    """
    # Build a machine learning pipeline
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('text_pipeline',Pipeline([
            ('vect', CountVectorizer(preprocessor=normalize_corpus)),
            ('tfidf', TfidfTransformer())
        ])),
        ('text_length',Text_Length_Extractor())
    ])),
        
    ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(random_state=42)))
    ])

    
    # Define a score used in scoring parameter
    def avg_accuracy(y_test, y_pred):
        """
        This is the score_func used in make_scorer, which would be used in in GridSearchCV 
        """
        avg_accuracy=accuracy_score(y_test.values.reshape(-1,1), y_pred.reshape(-1,1))
        return avg_accuracy
    avg_accuracy_cv = make_scorer(avg_accuracy)
    
    parameters = parameters = {
    #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),  
    #'clf__estimator__max_depth': [15, 30],  
    'clf__estimator__n_estimators': [100, 250]}
    
    # Use grid search to find better parameters.
    GridSearchCV_pipeline = GridSearchCV(pipeline, param_grid=parameters,cv=2,
                      scoring=avg_accuracy_cv, verbose=3)
    
    return GridSearchCV_pipeline

def evaluate_model(model, X_test, y_test,category_names):
    """
    The evaluate_model function will return the accuracy, precision, and recall, and f1 scores \
    for each output category of the dataset.

    INPUTS:
        model- a trained model for evaluation
        X_test - a panda data frame or Numpy array, contains the untouched values of features. 
        y_pred - a Numpy array, contains predicted category values of the messages. 
        
    OUTPUT:
        metrics_df, a panda dataframe that contains accuracy, precision, and recall, and f1 scores\
        for each output category of the dataset.
    """
    y_pred=model.predict(X_test)
    metrics_list_all=[]
    for col in range(y_test.shape[1]):
        accuracy = accuracy_score(y_test.iloc[:,col], y_pred[:,col])
        precision=precision_score(y_test.iloc[:,col], y_pred[:,col])
        recall = recall_score(y_test.iloc[:,col], y_pred[:,col])
        f_1 = f1_score(y_test.iloc[:,col], y_pred[:,col])
        metrics_list=[accuracy,precision,recall,f_1]
        metrics_list_all.append(metrics_list)
    metrics_df=pd.DataFrame(metrics_list_all,index=category_names,columns=
                            ["Accuracy","Precision","Recall","F_1"])
    print(metrics_df)
    print("----------------------------------------------------------------------")
    print(("The average accuracy score among all categories is {:.4f},\nthe average precision score score among all categories is {:.4f},\nthe average recall score among all categories is {:.4f},\nthe average F 1 score among all categories is {:.4f}").
          format(metrics_df.mean()["Accuracy"],metrics_df.mean()["Precision"],
                 metrics_df.mean()["Recall"],metrics_df.mean()["F_1"]))
    
    return None

def save_model(model, model_filepath):
    """
     Use the pickle operation to save the model to a file.
    INPUTS:
        model -
        model_filepath - A Python str object - the name of the model saved
    OUTPUT:
        None
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


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
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()