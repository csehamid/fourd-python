import pandas as pd
import numpy as np
from time import time
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

#loading data.csv as train data--------------------------
dataPath = './data.csv'
data = pd.read_csv(dataPath)

#tf idf vectorizing--------------------------------------------------
#ngram_range
#with no ngram the result are simple words
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,3), strip_accents='unicode', analyzer='word')
xData = tfidf.fit_transform(data['commit'])
yData = np.array(data['classifier'])



#RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(xData,yData)
test_prediction = clf.predict(xData)

#prediction
predicted = cross_val_predict(clf, xData, yData, cv=5)
predictionAccuracy=metrics.accuracy_score(yData, predicted)
print ("RandomForestClassifier Prediction Accuracy: %0.3f" %predictionAccuracy)


#Deciscion Tree
clf = tree.DecisionTreeClassifier()
clf.fit(xData,yData)
test_prediction = clf.predict(xData)

#prediction
predicted = cross_val_predict(clf, xData, yData, cv=5)
predictionAccuracy=metrics.accuracy_score(yData, predicted)
print ("Deciscion Tree Prediction Accuracy: %0.3f" %predictionAccuracy)


#Support Vector Machine SVC
clf = svm.SVC()
clf.fit(xData,yData)
test_prediction = clf.predict(xData)

#prediction
predicted = cross_val_predict(clf, xData, yData, cv=5)
predictionAccuracy=metrics.accuracy_score(yData, predicted)
print ("Support Vector Machine SVC Prediction Accuracy: %0.3f" %predictionAccuracy)

#linear SVC
clf = svm.LinearSVC()
clf.fit(xData,yData) 
test_prediction = clf.predict(xData)

#prediction
predicted = cross_val_predict(clf, xData, yData, cv=5)
predictionAccuracy=metrics.accuracy_score(yData, predicted)
print ("linear SVC Prediction Accuracy: %0.3f" %predictionAccuracy)

#Bernoulli Naive Baies
clf = BernoulliNB()
clf.fit(xData, yData)
test_prediction = clf.predict(xData)

#prediction
predicted = cross_val_predict(clf, xData, yData, cv=5)
predictionAccuracy=metrics.accuracy_score(yData, predicted)
print ("Bernoulli Naive Baies Prediction Accuracy: %0.3f" %predictionAccuracy)

#Multinomial Naive Baies
clf = MultinomialNB()
clf.fit(xData,yData)
test_prediction = clf.predict(xData)

#prediction
predicted = cross_val_predict(clf, xData, yData, cv=5)
predictionAccuracy=metrics.accuracy_score(yData, predicted)
print ("Multinomial Naive Baies Prediction Accuracy: %0.3f" %predictionAccuracy)