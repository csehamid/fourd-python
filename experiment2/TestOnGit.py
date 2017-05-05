from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import pickle
import sys
import math
#disabling warnings for some warnings like version and support warnings
import warnings
warnings.filterwarnings("ignore")


classifier_resampler=["RandomForestClassifier"
,"RandomForestClassifierRandomUnderSampler"
,"RandomForestClassifierTomekLinks"
,"RandomForestClassifierClusterCentroids"
,"RandomForestClassifierNearMiss"
,"RandomForestClassifierCondensedNearestNeighbour"
,"RandomForestClassifierOneSidedSelection"
,"RandomForestClassifierNeighbourhoodCleaningRule"
,"RandomForestClassifierEditedNearestNeighbours"
,"RandomForestClassifierInstanceHardnessThreshold"
,"RandomForestClassifierRepeatedEditedNearestNeighbours"
,"RandomForestClassifierRandomOverSampler"
,"RandomForestClassifierSMOTE"
,"RandomForestClassifierSMOTE borderline1"
,"RandomForestClassifierSMOTE borderline2"
,"RandomForestClassifierSMOTE svm"
,"RandomForestClassifierADASYN"
,"RandomForestClassifierSMOTETomek"
,"RandomForestClassifierSMOTEENN"
,"DecisionTreeClassifier"
,"DecisionTreeClassifierRandomUnderSampler"
,"DecisionTreeClassifierTomekLinks"
,"DecisionTreeClassifierClusterCentroids"
,"DecisionTreeClassifierNearMiss"
,"DecisionTreeClassifierCondensedNearestNeighbour"
,"DecisionTreeClassifierOneSidedSelection"
,"DecisionTreeClassifierNeighbourhoodCleaningRule"
,"DecisionTreeClassifierEditedNearestNeighbours"
,"DecisionTreeClassifierInstanceHardnessThreshold"
,"DecisionTreeClassifierRepeatedEditedNearestNeighbours"
,"DecisionTreeClassifierRandomOverSampler"
,"DecisionTreeClassifierSMOTE"
,"DecisionTreeClassifierSMOTE borderline1"
,"DecisionTreeClassifierSMOTE borderline2"
,"DecisionTreeClassifierSMOTE svm"
,"DecisionTreeClassifierADASYN"
,"DecisionTreeClassifierSMOTETomek"
,"DecisionTreeClassifierSMOTEENN"
,"SVC"
,"SVCRandomUnderSampler"
,"SVCTomekLinks"
,"SVCClusterCentroids"
,"SVCNearMiss"
,"SVCCondensedNearestNeighbour"
,"SVCOneSidedSelection"
,"SVCNeighbourhoodCleaningRule"
,"SVCEditedNearestNeighbours"
,"SVCInstanceHardnessThreshold"
,"SVCRepeatedEditedNearestNeighbours"
,"SVCRandomOverSampler"
,"SVCSMOTE"
,"SVCSMOTE borderline1"
,"SVCSMOTE borderline2"
,"SVCSMOTE svm"
,"SVCADASYN"
,"SVCSMOTETomek"
,"SVCSMOTEENN"
,"LinearSVC"
,"LinearSVCRandomUnderSampler"
,"LinearSVCTomekLinks"
,"LinearSVCClusterCentroids"
,"LinearSVCNearMiss"
,"LinearSVCCondensedNearestNeighbour"
,"LinearSVCOneSidedSelection"
,"LinearSVCNeighbourhoodCleaningRule"
,"LinearSVCEditedNearestNeighbours"
,"LinearSVCInstanceHardnessThreshold"
,"LinearSVCRepeatedEditedNearestNeighbours"
,"LinearSVCRandomOverSampler"
,"LinearSVCSMOTE"
,"LinearSVCSMOTE borderline1"
,"LinearSVCSMOTE borderline2"
,"LinearSVCSMOTE svm"
,"LinearSVCADASYN"
,"LinearSVCSMOTETomek"
,"LinearSVCSMOTEENN"
,"BernoulliNB"
,"BernoulliNBRandomUnderSampler"
,"BernoulliNBTomekLinks"
,"BernoulliNBClusterCentroids"
,"BernoulliNBNearMiss"
,"BernoulliNBCondensedNearestNeighbour"
,"BernoulliNBOneSidedSelection"
,"BernoulliNBNeighbourhoodCleaningRule"
,"BernoulliNBEditedNearestNeighbours"
,"BernoulliNBInstanceHardnessThreshold"
,"BernoulliNBRepeatedEditedNearestNeighbours"
,"BernoulliNBRandomOverSampler"
,"BernoulliNBSMOTE"
,"BernoulliNBSMOTE borderline1"
,"BernoulliNBSMOTE borderline2"
,"BernoulliNBSMOTE svm"
,"BernoulliNBADASYN"
,"BernoulliNBSMOTETomek"
,"BernoulliNBSMOTEENN"
,"NearestCentroid"
,"NearestCentroidRandomUnderSampler"
,"NearestCentroidTomekLinks"
,"NearestCentroidClusterCentroids"
,"NearestCentroidNearMiss"
,"NearestCentroidCondensedNearestNeighbour"
,"NearestCentroidOneSidedSelection"
,"NearestCentroidNeighbourhoodCleaningRule"
,"NearestCentroidEditedNearestNeighbours"
,"NearestCentroidInstanceHardnessThreshold"
,"NearestCentroidRepeatedEditedNearestNeighbours"
,"NearestCentroidRandomOverSampler"
,"NearestCentroidSMOTE"
,"NearestCentroidSMOTE borderline1"
,"NearestCentroidSMOTE borderline2"
,"NearestCentroidSMOTE svm"
,"NearestCentroidADASYN"
,"NearestCentroidSMOTETomek"
,"NearestCentroidSMOTEENN"]

#--------------------------my print to resolve encoding problems
def uprint(*objects, sep=' ', end='\n', file=sys.stdout):
    enc = file.encoding
    if enc == 'UTF-8':
        print(*objects, sep=sep, end=end, file=file)
    else:
        f = lambda obj: str(obj).encode(enc, errors='backslashreplace').decode(enc)
        print(*map(f, objects), sep=sep, end=end, file=file)

  
  

#-----------------load second data
#loading data 
dataPath = './git.csv'
data = pd.read_csv(dataPath)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,3), strip_accents='unicode', analyzer='word')
X = tfidf.fit_transform(data['commit'])
y_true = np.array(data['classifier'])
X=X.toarray()



# ---------------------------------------------------------------load classifier1
f1=open('./testResult.csv', 'w+')

#Printing header
uprint("classifier-resampler" ,    ","   ,"predictionAccuracy",  ","   ,"TP",  ","   ,"TN",  ","   ,"FP",  ","   ,"FN",  ","   ,"fmeasure",   ","   ,"precision",   ","   ,"recall",   ","   , "MatthewsCorrelationCoefficient",   ","   ,"kappa",   ","  ,"specificity",   ",","gmean",   ",","bacc",   ",", file=f1)
for index,value in enumerate(classifier_resampler):
    #print (index,value)
    #load classifier
    with open(value, 'rb') as f:
        clf = pickle.load(f)
    
    y_pred=clf.predict(X)
    predictionAccuracy=metrics.accuracy_score(y_true, y_pred)
    fmeasure=metrics.f1_score(y_true, y_pred)
    MatthewsCorrelationCoefficient=metrics.matthews_corrcoef(y_true, y_pred)
    kappa=metrics.cohen_kappa_score(y_true, y_pred)
    precision=metrics.precision_score(y_true, y_pred)
    #recall or sensitivity
    recall=metrics.recall_score(y_true, y_pred)   
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    specificity=TN/(TN+FP)
    gmean=math.sqrt(recall*specificity)
    bacc=0.5*recall+specificity
    
    print(predictionAccuracy)
    uprint(value ,    ","   ,predictionAccuracy,  ","   ,((TP*100)/y_true.size),  ","   ,((TN*100)/y_true.size),  ","   ,((FP*100)/y_true.size),  ","   ,((FN*100)/y_true.size),  ","   ,fmeasure,   ","   ,precision,   ","   ,recall,   ","   , MatthewsCorrelationCoefficient,   ","   ,kappa,   ","  ,specificity,   ",",gmean,   ",",bacc,   ",", file=f1)

f1.close()        
