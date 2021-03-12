import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn import preprocessing


def read_data():
  train_data = pd.read_csv('data/DBLPTrainset.csv')
  test_data = pd.read_csv('data/DBLPTestset.csv')
  test_data['Label'] = pd.read_csv('data/DBLPTestGroundTruth.csv')['Label']
  train_data['Label_Cat'] = train_data['Label'].astype('category').cat.codes
  test_data['Label_Cat'] = test_data['Label'].astype('category').cat.codes
  return train_data, test_data


def stem_tokenizer(text):
  stop_words = stopwords.words("english")
  tokens = nltk.word_tokenize(text)
  tokens = [word.lower() for word in tokens if word.isalpha()]
  filtered_words = [word for word in tokens if word not in stop_words]
  stemmer = PorterStemmer()
  porter_tokens = [stemmer.stem(word) for word in filtered_words]
  return porter_tokens

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
def lemma_tokenizer(text):
    stop_words = stopwords.words("english")
    tokens = TreebankWordTokenizer().tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    filtered_words = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in filtered_words]
    return lemmas

def extract(train, test):
  vector = CountVectorizer(ngram_range=(1,2), tokenizer=lemma_tokenizer)
  vector.fit(train.Title)
  return vector.transform(train.Title), vector.transform(test.Title)

  # tfidfconverter = TfidfVectorizer()
  # tfidfconverter.fit(train.Title)
  # return tfidfconverter.transform(train.Title), tfidfconverter.transform(test.Title)