#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

dataset = np.genfromtxt('car.data', delimiter=',', dtype=None)

n_samples = dataset.shape[0]
n_features = dataset.shape[1]
print("Car Evaluation dataset: %d amostras(%d características)" % (dataset.shape[0], n_features))

target = dataset[:,-1]
dataset = dataset[:,0:-1]

tle = LabelEncoder()
target = tle.fit_transform(target)
#print(np.unique(target))
#print(list(tle.classes_))
#exit()

labels_encoders = []
for idx in range(0, n_features-1):
    le = LabelEncoder()
    labels_encoders += [le]
    e = le.fit_transform(dataset[:,idx])
    dataset[:,idx] = e
    #print(list(le.classes_))
#print(len(dataset[0]))

dataset = dataset.astype(np.int32)

def get_indexes_feature_equals(idx_feature, value, rows=np.newaxis):
	x  = labels_encoders[idx_feature].transform([value])[0]
	if rows == None:
		return np.where(dataset[rows:,idx_feature] == x)[0]
	return np.where(dataset[rows,idx_feature] == x)[0]
	

x1medIdxs = get_indexes_feature_equals(0, 'med')
x1medCount = len(x1medIdxs)
print("P(x1=med) = %0.2f" % (float(x1medCount)/n_samples))

x2lowIdxs = get_indexes_feature_equals(1, 'low')
x2lowCount = len(x2lowIdxs)
print("P(x2=low) = %0.2f" % (float(x2lowCount)/n_samples))

x3is2Idxs = get_indexes_feature_equals(2, '2')
#print(x3is2Idxs)
x6HighIdxs = get_indexes_feature_equals(5, 'high', x3is2Idxs)
#print(x6HighIdxs)
print("P(x6=high|x3=2) = %0.2f" % (float(len(x6HighIdxs))/len(x3is2Idxs)))

enc=OneHotEncoder(sparse=False)
dataset = enc.fit_transform(dataset)
print(enc.active_features_)
print(enc.feature_indices_)

#print(dataset)
