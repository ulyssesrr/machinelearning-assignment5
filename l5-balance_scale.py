#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from blessings import Terminal

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

t = Terminal()

dataset = np.genfromtxt('balance-scale.data', delimiter=',', dtype=str)
#print(len(dataset))
n_samples = dataset.shape[0]
n_features = dataset.shape[1]
print(t.blue("Balance Scale dataset: %d amostras(%d características)" % (dataset.shape[0], n_features)))

def get_indexes_feature_equals(values_dict, rows = np.arange(len(dataset))):
	idxs = rows
	for idx_feature, value in values_dict.items():
		#x  = labels_encoders[idx_feature].transform([value])[0]
		x = value
		idxs = np.where(dataset[idxs,idx_feature] == x)[0]
	return idxs
	
def get_probability(value, given_value={}, rows = np.arange(len(dataset))):
	d = dict(list(value.items()) + list(given_value.items()))
	val = float(len(get_indexes_feature_equals(d, rows)))/len(get_indexes_feature_equals(given_value, rows))
	#print(len(get_indexes_feature_equals(d, rows)),len(get_indexes_feature_equals(given_value, rows)))
	str1 = ', '.join("{!s}={!r}".format(key,val) for (key,val) in value.items())
	str2 = ', '.join("{!s}={!r}".format(key,val) for (key,val) in given_value.items())
	#print("P(%s|%s) = %0.4f" % (str1, str2, val))
	return val

def naive_bayes(train_indexes, X, alpha=1):
	pred = []
	for cx in X:
		current_val = -1
		current_class = ""
		for t in ['B', 'L', 'R']:
			pc = get_probability({0: t}, rows=train_indexes)
			p1 = get_probability({1: cx[1]}, {0: t}, rows=train_indexes) + alpha
			p2 = get_probability({2: cx[2]}, {0: t}, rows=train_indexes) + alpha
			p3 = get_probability({3: cx[3]}, {0: t}, rows=train_indexes) + alpha
			p4 = get_probability({4: cx[4]}, {0: t}, rows=train_indexes) + alpha
			pc = pc*p1*p2*p3*p4
			#print(pc,p1,p2,p3,p4, pc*p1*p2*p3*p4)
			if pc > current_val:
				current_val = pc
				current_class = t
		pred += [current_class]
		#exit()
	return pred

hold_out = cross_validation.ShuffleSplit(n_samples, n_iter=10, test_size=.25)
accuracy_array = []
for train_indexes, test_indexes in hold_out:
	y_pred = naive_bayes(train_indexes, dataset[test_indexes], alpha=0)
	#print(y_pred)
	accuracy_test = accuracy_score(dataset[test_indexes,0], y_pred)
	accuracy_array += [accuracy_test]
#print("MEU NaiveBayes (Discreto): Acurácia (Média/Desvio Padrão): %0.2f/%0.2f" % (np.mean(accuracy_array), np.std(accuracy_array)))

target = dataset[:,0]
dataset = dataset[:,1:].astype(np.int32)

#print(np.unique(dataset[:,]))

print(t.green("3.A"))
hold_out = cross_validation.ShuffleSplit(n_samples, n_iter=10, test_size=.25)
accuracy_array = []
for train_indexes, test_indexes in hold_out:
	gnb = GaussianNB()
	gnb.fit(dataset[train_indexes], target[train_indexes])
	y_pred = gnb.predict(dataset[test_indexes])
	accuracy_test = accuracy_score(target[test_indexes], y_pred)
	accuracy_array += [accuracy_test]
print("NaiveBayes (Gaussiano): Acurácia (Média/Desvio Padrão): %0.2f/%0.2f" % (np.mean(accuracy_array), np.std(accuracy_array)))


print(t.green("3.B"))
#dataset[0] = [ 0,0,0,0]
#target[0] = 'B'

#enc=OneHotEncoder(sparse=False)
#dataset = enc.fit_transform(dataset)

hold_out = cross_validation.ShuffleSplit(n_samples, n_iter=10, test_size=.25)
accuracy_array = []

for train_indexes, test_indexes in hold_out:
	mnb = MultinomialNB(alpha=0, fit_prior=True)
	mnb.fit(dataset[train_indexes], target[train_indexes])
	y_pred = mnb.predict(dataset[test_indexes])
	accuracy_test = accuracy_score(target[test_indexes], y_pred)
	accuracy_array += [accuracy_test]
print("NaiveBayes (Discreto): Acurácia (Média/Desvio Padrão): %0.2f/%0.2f" % (np.mean(accuracy_array), np.std(accuracy_array)))


print(t.green("3.C"))

hold_out = cross_validation.ShuffleSplit(n_samples, n_iter=10, test_size=.25)
accuracy_array = []

for train_indexes, test_indexes in hold_out:
	mnb = MultinomialNB(alpha=1.0, fit_prior=True)
	mnb.fit(dataset[train_indexes], target[train_indexes])
	y_pred = mnb.predict(dataset[test_indexes])
	accuracy_test = accuracy_score(target[test_indexes], y_pred)
	accuracy_array += [accuracy_test]
print("NaiveBayes (Discreto+Suavizado): Acurácia (Média/Desvio Padrão): %0.2f/%0.2f" % (np.mean(accuracy_array), np.std(accuracy_array)))



exit()

## Testes abaixo
#print(dataset[0])


pe = get_probability({1: '1'}) * get_probability({2: '5'}) * get_probability({3: '4'}) * get_probability({4: '2'})

pc = get_probability({0: 'B'})
p1 = get_probability({1: '1'}, {0: 'B'})
p2 = get_probability({2: '5'}, {0: 'B'})
p3 = get_probability({3: '4'}, {0: 'B'})
p4 = get_probability({4: '2'}, {0: 'B'})
print("P(...|B) = %0.4f" % (pc*p1*p2*p3*p4/pe))

pc = get_probability({0: 'L'})
p1 = get_probability({1: '1'}, {0: 'L'})
p2 = get_probability({2: '5'}, {0: 'L'})
p3 = get_probability({3: '4'}, {0: 'L'})
p4 = get_probability({4: '2'}, {0: 'L'})
print("P(...|L) = %0.4f" % (pc*p1*p2*p3*p4/pe))

pc = get_probability({0: 'R'})
p1 = get_probability({1: '1'}, {0: 'R'})
p2 = get_probability({2: '5'}, {0: 'R'})
p3 = get_probability({3: '4'}, {0: 'R'})
p4 = get_probability({4: '2'}, {0: 'R'})
print("P(...|R) = %0.4f" % (v/pe))

print("P(C=L|x1,x2,x3,x4=1) = %0.2f" % get_probability({0: 'L'}, {1: '1',2: '1',3: '1',4: '1'}))
print("P(C=R|x1,x2,x3,x4=1) = %0.2f" % get_probability({0: 'R'}, {1: '1',2: '1',3: '1',4: '1'}))
