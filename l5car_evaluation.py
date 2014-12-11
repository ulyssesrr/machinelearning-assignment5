#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from blessings import Terminal

import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

t = Terminal()

dataset = np.genfromtxt('car.data', delimiter=',', dtype=None)

n_samples = dataset.shape[0]
n_features = dataset.shape[1]
print(t.blue("Car Evaluation dataset: %d amostras(%d características)" % (dataset.shape[0], n_features)))

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

def get_indexes_feature_equals(values_dict):
	rows = np.arange(len(dataset))
	for idx_feature, value in values_dict.items():
		x  = labels_encoders[idx_feature].transform([value])[0]
		rows = np.where(dataset[rows,idx_feature] == x)[0]
	return rows
	
def get_probability(value, given_value={}):
	d = dict(list(value.items()) + list(given_value.items()))
	return float(len(get_indexes_feature_equals(d)))/len(get_indexes_feature_equals(given_value))

print(t.green("1.A"))
print("P(x1=med) = %0.2f" % get_probability({0: 'med'}))

print("P(x2=low) = %0.2f" % get_probability({1: 'low'}))

print(t.green("1.B"))

print("P(x6=high|x3=2) = %0.2f" % get_probability({5: 'high'}, {2: '2'}))
print("P(x2=low|x4=4) = %0.2f" % get_probability({1: 'low'}, {3: '4'}))

print(t.green("1.C"))

print("P(x1=low|x2=low,X5=small) = %0.2f" % get_probability({0: 'low'}, {1: 'low', 4: 'small'}))
print("P(x4=4|x1=med,x3=2) = %0.2f" % get_probability({3: '4'}, {0: 'med', 2: '2'}))

print(t.green("1.D"))

print("P(x2= vhigh,x3=2|x4=2) = %0.2f" % get_probability({1: 'vhigh'}, {2: '2', 3: '2'}))
print("P(x3=4,x5=med|x1=med) = %0.2f" % get_probability({2: '4', 4: 'med'}, {0: 'med'}))

enc=OneHotEncoder(sparse=False)
dataset = enc.fit_transform(dataset)
print(enc.active_features_)
print(enc.feature_indices_)

#print(dataset)
