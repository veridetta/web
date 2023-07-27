import pandas as pd
import numpy as np
place_data = pd.read_csv("dataset (1).csv")
place_data.head()

#case folding
place_data['description'] = place_data['description'].str.lower()
print('Case folding result : \n')
print(place_data['description'].head(5))
print('\n\n\n')

import string
import re #regex library
import nltk
#import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
nltk.download('punkt')
#Tokenizing

def remove_place_special(text):
  if isinstance(text, str):
    #remove tab, new line, and back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ")
    #remove non ASCII (emoticon, chinese word, etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    #remove mention, link, hastag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    #remove incomplete url
    return text.replace("http://", " ").replace("https://", " ")

place_data['description'] = place_data['description'].apply(remove_place_special)

#remove number
def remove_number(text):
  if isinstance(text, str):
    return re.sub(r"\d+", "", text)

place_data['description'] = place_data['description'].apply(remove_number)

#remove punctuation
def remove_punctuation(text):
  if isinstance(text, str):
    return text.translate(str.maketrans("","",string.punctuation))

place_data['description'] = place_data['description'].apply(remove_punctuation)

#remove whitespace leading & trailing
def remove_whitespace_LT(text):
  if isinstance(text, str):
    return text.strip()

place_data['description'] = place_data['description'].apply(remove_whitespace_LT)

#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
  if isinstance(text, str):
    return re.sub('\s+',' ',text)

place_data['description'] = place_data['description'].apply(remove_whitespace_multiple)

#remove single char
def remove_single_char(text):
  if isinstance(text, str):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

place_data['description'] = place_data['description'].apply(remove_single_char)

#nltk word tokenize
def word_tokenize_wrapper(text):
  if isinstance(text, str):
    return word_tokenize(text)

place_data['place_tokens'] = place_data['description'].apply(word_tokenize_wrapper)

print('Tokenixing Result : \n')
print(place_data['place_tokens'].head())
print('\n\n\n')

#nltk calc frequency distribution
def freqDist_wrapper(text):
  return FreqDist(text)

place_data['place_tokens_fdist'] = place_data['place_tokens'].apply(freqDist_wrapper)

print('Frequency tokens : \n')
print(place_data['place_tokens_fdist'].head().apply(lambda x : x.most_common()))


from nltk.corpus import stopwords
nltk.download("stopwords")
#get stopword from nltk stopword
#get stopword indonesia
list_stopwords = stopwords.words('indonesian')

#convert list to dictionary
list_stopwords = set(list_stopwords)

#remove stopword pada list token
def stopwords_removal(words):
  if words is not None:
    return [word for word in words if word not in list_stopwords]
  else:
    return []

place_data['place_tokens_WSW'] = place_data['place_tokens'].apply(stopwords_removal)

print(place_data['place_tokens_WSW'].head())


!pip3 install swifter
!pip3 install PySastrawi


#import sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter

#create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#stemmed
def stemmed_wrapper(term):
  return stemmer.stem(term)

term_dict = {}

for document in place_data['place_tokens_WSW']:
  for term in document:
    if term not in term_dict:
      term_dict[term] = ' '

print(len(term_dict))
print("-------------------")

for term in term_dict:
  term_dict[term] = stemmed_wrapper(term)
  print(term,":" ,term_dict[term])

print(term_dict)
print("-------------------")

#apply stemmed term to dataframe
def get_stemmed_term(document):
  return [term_dict[term] for term in document]

place_data['place_tokens_stemmed'] = place_data['place_tokens_WSW'].swifter.apply(get_stemmed_term)
print(place_data['place_tokens_stemmed'])

!pip3 install -U scikit-learn

import pandas as pd
import numpy as np

place_data = pd.read_csv("hasil_processing.csv", usecols=["id", "place_name", "place_tokens_stemmed"])
place_data.columns = ["id", "place_name", "place"]

place_data.head()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfv = TfidfVectorizer (min_df=3, max_features=None,
                     strip_accents='unicode', analyzer='word', token_pattern=r'\w+',
                     ngram_range=(1, 10),
                     stop_words = 'english')

place_data['place'] = place_data['place'].fillna('')
tfv_matrix = tfv.fit_transform(place_data['place'])

cosine_sim = cosine_similarity(tfv_matrix, tfv_matrix)

indices = pd.Series(place_data.index, index=place_data['place_name']).drop_duplicates()

list(enumerate(cosine_sim[indices['Tahu Petis Kertasari ']]))

sorted(list(enumerate(cosine_sim[indices['Tahu Petis Kertasari ']])), key=lambda x: x[1], reverse=True)

def give_rec(place_name, cosine_sim=cosine_sim):
    idx = indices[place_name]
    cosine_sim_scores = list(enumerate(cosine_sim[idx]))
    cosine_sim_scores = sorted(cosine_sim_scores, key=lambda x: x[1], reverse=True)
    cosine_sim_scores = cosine_sim_scores[1:6]
    place_data_indices = [i[0] for i in cosine_sim_scores]
    return place_data[['place_name', 'place']].iloc[place_data_indices]