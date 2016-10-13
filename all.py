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

#loading got.csv as train data--------------------------
pathtrain = './git.csv'
rawTrain = pd.read_csv(pathtrain)

#loading forge.csv as test data--------------------------
pathtest = './forge.csv'
rawTest = pd.read_csv(pathtest)


#tf idf vectorizing--------------------------------------------------
#ngram_range
#with no ngram the result are simple words
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,3), strip_accents='unicode', analyzer='word')
xTrain = tfidf.fit_transform(rawTrain['commit'])

yTrain = np.array(rawTrain['classifier'])

xTest=tfidf.fit_transform(rawTest['commit'])
yTest = np.array(rawTest['classifier'])


#RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(xTrain,yTrain)
test_prediction = clf.predict(xTest)

print("\nRandomForestClassifier :")
score1 = metrics.accuracy_score(test_prediction,yTrain)
print("accuracy prediction and train set:   %0.3f" % score1)
score1 = metrics.accuracy_score(test_prediction,yTest)
print("accuracy prediction and test set:   %0.3f" % score1)


#Deciscion Tree
clf = tree.DecisionTreeClassifier()
clf.fit(xTrain,yTrain)
test_prediction = clf.predict(xTest)

print("\nDecision Tree")
score1 = metrics.accuracy_score(test_prediction,yTrain)
print("accuracy prediction and train set:   %0.3f" % score1)
score1 = metrics.accuracy_score(test_prediction,yTest)
print("accuracy prediction and test set:   %0.3f" % score1)


#Support Vector Machine SVC
clf = svm.SVC()
clf.fit(xTrain,yTrain)
test_prediction = clf.predict(xTest)

print("\nSVM.SVC")
score1 = metrics.accuracy_score(test_prediction,yTrain)
print("accuracy prediction and train set:   %0.3f" % score1)
score1 = metrics.accuracy_score(test_prediction,yTest)
print("accuracy prediction and test set:   %0.3f" % score1)

#linear SVC
clf = svm.LinearSVC()
clf.fit(xTrain,yTrain) 
test_prediction = clf.predict(xTest)

print("\nLinearSVC")
score1 = metrics.accuracy_score(test_prediction,yTrain)
print("accuracy prediction and train set:   %0.3f" % score1)
score1 = metrics.accuracy_score(test_prediction,yTest)
print("accuracy prediction and test set:   %0.3f" % score1)

#Bernoulli Naive Baies
clf = BernoulliNB()
clf.fit(xTrain, yTrain)
test_prediction = clf.predict(xTest)

print("\nBernoulliNB")
score1 = metrics.accuracy_score(test_prediction,yTrain)
print("accuracy prediction and train set:   %0.3f" % score1)
score1 = metrics.accuracy_score(test_prediction,yTest)
print("accuracy prediction and test set:   %0.3f" % score1)

#Multinomial Naive Baies
clf = MultinomialNB()
clf.fit(xTrain,yTrain)
test_prediction = clf.predict(xTest)

print("\nMultinomialNB")
score1 = metrics.accuracy_score(test_prediction,yTrain)
print("accuracy prediction and train set:   %0.3f" % score1)
score1 = metrics.accuracy_score(test_prediction,yTest)
print("accuracy prediction and test set:   %0.3f" % score1)