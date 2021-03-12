from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import feature_extraction
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score, precision_score, recall_score

def classify(train, test):
  vect1, vect2 = feature_extraction.extract(train, test)
  classifiers = [MultinomialNB(), SGDClassifier(random_state=0)]

  for classifier in classifiers:
    predict = cross_val_predict(classifier, vect1, train.Label_Cat)
    print(predict)
    print(precision_score(train.Label_Cat, predict, average='micro'))
    print(recall_score(train.Label_Cat, predict, average='micro'))
    print(f1_score(train.Label_Cat, predict, average='micro'))

    print(classification_report(train.Label_Cat, predict))
    print(accuracy_score(train.Label_Cat, predict))

    print('on test data')
    classifier.fit(vect1, train.Label_Cat)
    y_pred = classifier.predict(vect2)
    print(classification_report(test.Label_Cat, y_pred))
    print(accuracy_score(test.Label_Cat, y_pred))

