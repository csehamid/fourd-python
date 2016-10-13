import pandas as pd
import numpy as np
from time import time
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

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

#print the csv
#print (rawTrain) #display the data

#print the actual word result of the vectorization
#print (tfidf.get_feature_names())

#print the voctorization result as numbers
#print (xTrain)
#print(yTrain)


xTest=tfidf.fit_transform(rawTest['commit'])
yTest = np.array(rawTest['classifier'])

clf = RandomForestClassifier()
clf.fit(xTrain,yTrain)





test_prediction = clf.predict(xTest)

score1 = metrics.accuracy_score(test_prediction,yTrain)
print("accuracy prediction and train set:   %0.3f" % score1)

score1 = metrics.accuracy_score(test_prediction,yTest)
print("accuracy prediction and test set:   %0.3f" % score1)


#Root-mean-square error
rmse=(np.sqrt(np.sum(np.array(np.array(clf.predict(xTrain))-yTrain)**2)/ (xTrain.shape[0]*24.0)))
print ("RMSE:   %0.3f" % rmse)


