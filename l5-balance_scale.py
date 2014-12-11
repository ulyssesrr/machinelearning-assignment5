#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from blessings import Terminal

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

t = Terminal()

dataset = np.genfromtxt('balance-scale.data', delimiter=',', dtype=str)
print(len(dataset))
n_samples = dataset.shape[0]
n_features = dataset.shape[1]
print(t.blue("Car Evaluation dataset: %d amostras(%d características)" % (dataset.shape[0], n_features)))

target = dataset[:,0]
dataset = dataset[:,1:].astype(np.int32)

print(np.unique(dataset[:,]))

print(t.green("3.A"))
hold_out = cross_validation.ShuffleSplit(n_samples, n_iter=10, test_size=.25)
accuracy_array = []
for train_indexes, test_indexes in hold_out:
	gnb = GaussianNB()
	gnb.fit(dataset[train_indexes], target[train_indexes])
	y_pred = gnb.predict(dataset[test_indexes])
	accuracy_test = accuracy_score(target[test_indexes], y_pred)
	accuracy_array += [accuracy_test]
print("NB (Gaussiana): Acurácia (Média/Desvio Padrão): %0.2f/%0.2f" % (np.mean(accuracy_array), np.std(accuracy_array)))


print(t.green("3.B"))
#dataset[0] = [ 0,0,0,0]
#target[0] = 'B'

enc=OneHotEncoder(sparse=False)
dataset = enc.fit_transform(dataset)

hold_out = cross_validation.ShuffleSplit(n_samples, n_iter=10, test_size=.25)
accuracy_array = []

mnb = BernoulliNB(alpha=0, fit_prior=True)
mnb.fit(dataset[train_indexes], target[train_indexes])
print(mnb.predict_proba(dataset[0]))
for train_indexes, test_indexes in hold_out:
	mnb = MultinomialNB(alpha=0, fit_prior=True)
	mnb.fit(dataset[train_indexes], target[train_indexes])
	y_pred = mnb.predict(dataset[test_indexes])
	accuracy_test = accuracy_score(target[test_indexes], y_pred)
	accuracy_array += [accuracy_test]
print("NB (Discreto): Acurácia (Média/Desvio Padrão): %0.2f/%0.2f" % (np.mean(accuracy_array), np.std(accuracy_array)))
