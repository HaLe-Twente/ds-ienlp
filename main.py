import feature_extraction
import classifier

# nltk.download('stopwords')
# nltk.download('wordnet')

train, test = feature_extraction.read_data()
# classifier.classify(train, test)
classifier.classify(train, test)