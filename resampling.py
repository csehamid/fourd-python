#December 2016
#Author: Hamid Shekarforoush sshekar@bgsu.edu
#
#
#this program read the data from a csv file and then reasmple it using diffrent resampling methods using
#libraries from (https://github.com/scikit-learn-contrib/imbalanced-learn)
#and feed it to diffrent classifiers using libraries from (http://scikit-learn.org/stable/)
#
#the results are discussed in the paper "Classifying Commit Messages: A Case Study in Resampling Techniques"
#
#



import pandas as pd
import numpy as np
import timeit

#importing classifiers metrics
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict

#importing classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import MultinomialNB

#importing samplers

#under samplers
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks

#over samplers
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

#combine samplers
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek

#disabling warnings for some warnings like version and support warnings
import warnings
warnings.filterwarnings("ignore")




#the list of classifiers
#MultinomialNB classifier is producing errors with some resamplers, because of negative value.
#its not compatible with all resamplers so it should be run separatly
classifiersList = ["RandomForestClassifier",
"DecisionTreeClassifier",
"SVC",
"LinearSVC",
"BernoulliNB",
"NearestCentroid"
#"MultinomialNB",
]

#the list of resamplers
samplerList = ["RandomUnderSampler",
"TomekLinks",
"ClusterCentroids",
"NearMiss",
"CondensedNearestNeighbour",
"OneSidedSelection",
"NeighbourhoodCleaningRule",
"EditedNearestNeighbours",
"InstanceHardnessThreshold",
"RepeatedEditedNearestNeighbours",
"RandomOverSampler",
"SMOTE",
"SMOTE(kind='borderline1')",
"SMOTE(kind='borderline2')",
"SMOTE(kind='svm')",
"ADASYN",
"SMOTETomek",
"SMOTEENN"]
#-------------------------------functions------------------------


#-------------------------------call all classifiers------------------------

#calling all classifiers in the order they listed in classifiersList
#index is the name of classifier
def callAll(xTemp,yTemp):
    commentCounter(yTemp)
    for index in (classifiersList):
        Classifiers(index , xTemp , yTemp)
     
#-------------------------------count comment------------------------

#counting the number of features in each class
def commentCounter( tmpData ):
    designComment=0
    normalComment=0
    for index in (tmpData):
        if index==1:
            designComment=designComment+1
        else:
            normalComment=normalComment+1

    print(designComment, end=" , ")
    print(normalComment, end=" , ")

#-------------------------------resamplers------------------------     


#-------------------------------Classifier------------------------    
def Classifiers(classifierName , xTemp , yTemp):
    
    #creating the classifier and fitting the data for it
    #globals()[classifierName] is converting the classifierName which is string to part of code, we use this to reduce the amount of code
    clf = globals()[classifierName]()
    clf.fit(xTemp,yTemp)
    
    #start time for the start prediction process
    start_time = timeit.default_timer()
    
    #prediction using 10 k fold
    predicted = cross_val_predict(clf, xTemp, yTemp, cv=10)
    
    #stop time for the end prediction process
    elapsed = timeit.default_timer() - start_time

#measring diffrent metrics, although we did not use all of them in the paper, we just calculate them in case they come in handy
    predictionAccuracy=metrics.accuracy_score(yTemp, predicted)
    fmeasure=metrics.f1_score(yTemp, predicted)
    MatthewsCorrelationCoefficient=metrics.matthews_corrcoef(yTemp, predicted)
    kappa=metrics.cohen_kappa_score(yTemp, predicted)
    precision=metrics.precision_score(yTemp, predicted)
    recall=metrics.recall_score(yTemp, predicted)
    TN, FP, FN, TP = confusion_matrix(yTemp, predicted).ravel()
    
#the next code section is all printing
#we use "," after each value so that hte output can be saved in a csv file for easy readability 

    #print classifier name
    print(classifierName, end=" , ")
    
    #print the time it takes to calculate prediction
    print(elapsed, end=" , ")
    
    #print prediction acccuracy with 10k fold
    print(predictionAccuracy, end=" , ")

    #printing the TP%
    print(((TP*100)/yTemp.size), end=" , ")
    
    #printing the TN%
    print(((TN*100)/yTemp.size), end=" , ")
    
    #printing the FP%
    print(((FP*100)/yTemp.size), end=" , ")
    
    #printing the FN%
    print(((FN*100)/yTemp.size), end=" , ")
    
    #print F-measure
    print(fmeasure, end=" , ")
    
    #print precision
    print(precision, end=" , ")
    
    #print recall
    print(recall, end=" , ")
    
    #print MatthewsCorrelationCoefficient
    print(MatthewsCorrelationCoefficient, end=" , ")
    
    #print kappa
    print(kappa, end=" , ")

    #just an empty space after each classifier, for easy readibility in csv file
    print(" " , end=" , ")


#-------------------------------end of functions------------------------    

# this section is the main part of program

#loading data.csv as train data--------------------------
dataPath = './data.csv'
data = pd.read_csv(dataPath)

#tf idf vectorizing
#with no ngram the result are simple words
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,3), strip_accents='unicode', analyzer='word')
xData = tfidf.fit_transform(data['commit']).toarray()
yData = np.array(data['classifier'])


#calling all classifiers on normal data without resampling as refrence data
print("no sampling" , end=" , ")
callAll(xData,yData)

resampleMethodName=""

#loop through the samplerList to call all resampling
#after each resampling method called, we call all classifiers on that resampling method
#locals()[samplerList [resampleMethod]]() doing the same thing as globals does in the classifiers function
#because borderline1, borderline 2 and svm can not be called directly using the locals we used 3 if, others will call normally
for resampleMethod in range (0,18) :
    
    #3 if for the 3 special cases
    if (resampleMethod==12):
        sm=SMOTE(kind='borderline1')
        resampleMethodName="SMOTE borderline1"
    elif (resampleMethod==13):
        sm=SMOTE(kind='borderline2')
        resampleMethodName="SMOTE borderline2"
    elif (resampleMethod==14):
        sm=SMOTE(kind='svm')
        resampleMethodName="SMOTE svm"
    else:
        sm=locals()[samplerList [resampleMethod]]()
        resampleMethodName=samplerList [resampleMethod]

    #start time for resampling
    start_time = timeit.default_timer()
    
    #feeding the data to resamplers to create new sample space
    x_resampled, y_resampled = sm.fit_sample(xData, yData)
    
    #end time for resampling
    elapsed = timeit.default_timer() - start_time
    #printing next line for compatibility with csv format
    print();

    #print resampling method name
    print (resampleMethodName, end=" , ")
    
    #print time it takes to resample
    print(elapsed, end=" , ")
    
    #calling all classifiers on the resampled data
    callAll(x_resampled,y_resampled)
    

