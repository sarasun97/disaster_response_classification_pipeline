import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk import word_tokenize

class Text_Length_Extractor(BaseEstimator, TransformerMixin):
    def get_length(self, text):
        """
        INPUT:
            text - Python str object - the raw text without cleaning
        OUTPUT:
            length - Python int object - the number of tokens in the text 
        """
        length=len(word_tokenize(text))
        return length
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        INPUT:
            X - A NumPy array or pd series, contains a series of messages used for classification
        OUTPUT:
            pd.DataFrame(X_length) - A panda data frame, - contains the number of tokens in the text 
        """
        X_length = pd.Series(X).apply(self.get_length)
        # In order to use FeatureUnion to combine the Text_Length_Extractor with the text_pipeline,
        # We must convert X_length into a dataframe. Otherwise, ValueError: blocks[0,:] has incompatible row dimensions. 
        return pd.DataFrame(X_length)