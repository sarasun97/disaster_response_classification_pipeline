#module for supressing warnings about future changes in Python:
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys

#html parser for digesting text taken from web-pages
import html.parser
from html.parser import HTMLParser

#module for handling regular expressions and special characters
import re

#Natural Language ToolKit
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

#modules for handling unicode characters and strings
import unicodedata
import string
import pattern3

import numpy as np
import pandas as pd

# Remove accents from characters
def normalize_accented_characters(text):
    """
    INPUT:
        text - Python str object - the raw text
    OUTPUT:
        text-  Python str object - the text after removing accents from characters
    """
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf8')
    return text

# Clean up HTML markups
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ' '.join(self.fed)
    
def strip_html(text):
    """
    INPUT:
        text - Python str object - the raw text
    OUTPUT:
        text-  Python str object - the text after cleaning up HTML markups
    """
    html_stripper = MLStripper()
    html_stripper.feed(text)
    return html_stripper.get_data()

# Expand contractions
# Defining the list of word contractions
contraction_mapping = {"ain't": "is not","aren't": "are not","can't": "cannot","can't've": "cannot have",
                       "'cause": "because","could've": "could have","couldn't": "could not",
                       "couldn't've": "could not have","didn't": "did not", "doesn't": "does not",
                       "don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not",
                       "haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will",
                       "he'll've":  "he he will have","he's": "he is","how'd": "how did","how'd'y": "how do you",
                       "how'll": "how will","how's": "how is", "I'd": "I would","I'd've": "I would have",
                       "I'll": "I will","I'll've": "I will have","I'm": "I am","I've": "I have","i'd": "i would",
                       "i'd've": "i would have","i'll": "i will","i'll've": "i will have","i'm": "i am",
                       "i've": "i have","isn't": "is not","it'd": "it would","it'd've": "it would have",
                       "it'll": "it will", "it'll've": "it will have","it's": "it is","let's": "let us",
                       "ma'am": "madam","mayn't": "may not", "might've": "might have","mightn't": "might not",
                       "mightn't've": "might not have","must've": "must have", "mustn't": "must not",
                       "mustn't've": "must not have","needn't": "need not","needn't've": "need not have",
                       "o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have",
                       "shan't": "shall not", "sha'n't": "shall not","shan't've": "shall not have","she'd": "she would",
                       "she'd've": "she would have", "she'll": "she will","she'll've": "she will have","she's": "she is",
                       "should've": "should have","shouldn't": "should not", "shouldn't've": "should not have",
                       "so've": "so have","so's": "so as","that'd": "that would","that'd've": "that would have",
                       "that's": "that is","there'd": "there would","there'd've": "there would have","there's": "there is",
                       "they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have",
                       "they're": "they are","they've": "they have", "to've": "to have","wasn't": "was not","we'd": "we would",
                       "we'd've": "we would have","we'll": "we will","we'll've": "we will have", "we're": "we are",
                       "we've": "we have","weren't": "were not","what'll": "what will","what'll've": "what will have",
                       "what're": "what are", "what's": "what is","what've": "what have","when's": "when is","when've": "when have",
                       "where'd": "where did","where's": "where is","where've": "where have","who'll": "who will",
                       "who'll've": "who will have","who's": "who is","who've": "who have","why's": "why is", "why've": "why have",
                       "will've": "will have","won't": "will not","won't've": "will not have","would've": "would have",
                       "y'all'd've": "you all would have","y'all're": "you all are", "y'all've": "you all have","you'd": "you would",
                       "you'd've": "you would have","you'll": "you will","you'll've": "you will have", "you're": "you are","you've": "you have"}

# Expand_contractions based on the list
def expand_contractions(text, contraction_mapping):
    """
    INPUT:
        text - Python str object - the raw text
        contraction_mapping- Python dictionary object - used to find expanded versions of contraction
       
    OUTPUT:
        expanded_text-  Python str object - the text after expanding contraction
    """
    contractions_pattern =re.compile('({})'.format('|'.join(contraction_mapping.keys())),                                     flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        """
        INPUT:
            contraction - Python str object - the contraction that needs to be expanded
       
        OUTPUT:
            expanded_contraction-  Python str object - the expanded contraction
        """
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# Tokenize the text
nltk.download('punkt')
def tokenize_text(text):
    """
    INPUT:
        text - Python str object - the text used for analysis
    OUTPUT:
        tokens-  Python list object - a list of tokens from the text
    """
    tokens = nltk.word_tokenize(text) 
    tokens = [token.strip() for token in tokens]
    return tokens

# Annotate text tokens with Part-Of-Speach tags
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def pos_tag_text(text_tokens):
    """
    INPUT:
        text_tokens - Python list object - a list of tokens from the text
       
    OUTPUT:
        tagged_lower_text- Python list object - a list of tokens with Part-Of-Speach tags
    """
    def penn_to_wn_tags(pos_tag):
        """
        INPUT:
            pos_tag- Python str object - Penn Treebank P.O.S. Tags
        OUTPUT:
            pos_tag- Python str object - WordNet P.O.S. Tags
        """
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None  
    tagged_text = nltk.pos_tag(text_tokens)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text

# Lemmatize text based on Part-Of-Speach (POS) tags
def lemmatize_text(text):
    """
    INPUT:
        text - Python str object - the raw text
    OUTPUT:
        text-  Python str object - the text after lemmatization
    """
    wnl = WordNetLemmatizer()
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                         else word                     
                         for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

# Remove special characters, such as punctuation marks
def remove_special_characters(text):
    """
    INPUT:
        text - Python str object - the raw text
    OUTPUT:
        text-  Python str object - the text after removing special characters
    """
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text 

# Remove all non-text characters (numbers, etc.)
def keep_text_characters(text):
    """
    INPUT:
        text - Python str object - the raw text
    OUTPUT:
        text-  Python str object - the text after removing all non-text characters 
    """
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# Get rid of stopwords
# Select the list of stopwords from NLTK and amend it by adding more stopwords to it
from nltk.corpus import stopwords
nltk.download('stopwords')
stopword_list = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    """
    INPUT:
        text - Python str object - the raw text
    OUTPUT:
        text-  Python str object - the text after removing stopwords
    """
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def normalize_corpus(text, only_text_chars=True): 
    """
    INPUT:
        text - Python str object - the raw text
        only_text_chars - Python boolean object - whether to remove numbers and only keep text
    OUTPUT:
        text-  Python str object - the text after all the preprocessing steps
    """
    text = normalize_accented_characters(text)
    text = html.unescape(text)
    text = strip_html(text)
    text = expand_contractions(text, contraction_mapping)
    text = tokenize_text(text)
    text = lemmatize_text(text)
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    if only_text_chars:
        text = keep_text_characters(text)
    return text