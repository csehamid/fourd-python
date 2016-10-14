import pandas as pd
import numpy as np
from time import time
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit


#loading got.csv as train data--------------------------
dataPath = './data.csv'
data = pd.read_csv(dataPath)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,3), strip_accents='unicode', analyzer='word')
xData = tfidf.fit_transform(data['commit'])
yData = np.array(data['classifier'])

#http://scikit-learn.org/stable/modules/cross_validation.html
#------------------cross validation with StratifiedKFold---------------------------------------------------
clf = RandomForestClassifier()
clf.fit(xData,yData)
#the defult cv uses KFold or StratifiedKFold strategies by default
scores = cross_val_score(clf, xData, yData, cv=5)
print("\n StratifiedKFold")
print (scores)
print ("average: %0.3f" %(sum(scores) / float(len(scores))))
#std :  standard deviation
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#prediction
predicted = cross_val_predict(clf, xData, yData, cv=5)
predictionAccuracy=metrics.accuracy_score(yData, predicted)
print ("Prediction Accuracy: %0.3f" %predictionAccuracy)

#------------------ end of cross validation  with StratifiedKFold---------------------------------------------------

#------------------cross validation with ShuffleSplit---------------------------------------------------
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
scores = cross_val_score(clf, xData, yData, cv=cv)
print("\n ShuffleSplit")
print (scores)
print ("average: %0.3f" %(sum(scores) / float(len(scores))))
#std :  standard deviation
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#prediction
#predicted = cross_val_predict(clf, xData, yData, cv=cv)
#predictionAccuracy=metrics.accuracy_score(yData, predicted)
#print ("Prediction Accuracy with shufflesplit: %0.3f" %predictionAccuracy)
#------------------end of cross validation with ShuffleSplit---------------------------------------------------

#split into 1 and 4 5times the train data are orderd and not randomized
#train1 1-400
#train2 401-800 ...
#skf = StratifiedKFold(n_splits=5)
#for test, train in skf.split(xData,yData):
    
#       print ("train %0.3i"  %train.size)
#       print ("test %0.3i"  %test.size)
#       print (xData.train)

#        clf = RandomForestClassifier()
#        clf.fit(xData.train,yData.train)
#        test_prediction = clf.predict(xData.test)
#        
#        print("\nRandomForestClassifier :")
#        score1 = metrics.accuracy_score(test_prediction,yData.test)
#        print("accuracy prediction and test set:   %0.3f" % score1)
        
        #Deciscion Tree
#        clf = tree.DecisionTreeClassifier()
#        clf.fit(xData,yData)
#        test_prediction = clf.predict(xData)
#        
#        print("\nDecision Tree")
#        score1 = metrics.accuracy_score(test_prediction,yData)
#        print("accuracy prediction and train set:   %0.3f" % score1)

