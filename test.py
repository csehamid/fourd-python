import pandas as pd
import numpy as np
from time import time
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


from sklearn.cross_validation import StratifiedKFold


#loading got.csv as train data--------------------------
dataPath = './data.csv'
data = pd.read_csv(dataPath)

print (data)


tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,3), strip_accents='unicode', analyzer='word')
xData = tfidf.fit_transform(data['commit'])
yData = np.array(data['classifier'])

skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(data):
        print("%s %s" % (train, test))
