import mlrose_hiive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from ucimlrepo import fetch_ucirepo 
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def ga_clbr(input_tuple):

	(pop_size, mutation_prob, max_iters, learning_rate, early_stopping, max_attempts, seed, 
     X_train, y_train, X_vld, y_vld, output_file) = input_tuple
	print(pop_size, mutation_prob, max_iters, learning_rate, early_stopping, max_attempts, seed, 
    file=open(output_file+'_progress.txt', 'a'))

	nn_ga = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation = 'relu', \
	                                   algorithm = 'genetic_alg', max_iters = max_iters, \
	                                   bias = True, is_classifier = True, learning_rate = learning_rate, \
	                                   early_stopping = early_stopping, clip_max = 5.0, max_attempts = max_attempts, \
	                                   random_state = seed, pop_size=pop_size, mutation_prob=mutation_prob, curve=True)
	tic = time.perf_counter()
	nn_ga.fit(X_train, y_train)
	toc = time.perf_counter()
	train_time = toc-tic
	# train
	y_train_pred = nn_ga.predict(X_train)
	auc_train = roc_auc_score(y_train, nn_ga.predicted_probs)
	acc_train = accuracy_score(y_train, y_train_pred)
	f1_train = f1_score(y_train, y_train_pred)
	# validation
	y_vld_pred = nn_ga.predict(X_vld)
	auc_vld = roc_auc_score(y_vld, nn_ga.predicted_probs)
	acc_vld = accuracy_score(y_vld, y_vld_pred)
	f1_vld = f1_score(y_vld, y_vld_pred)

	print(pop_size, mutation_prob, max_iters, learning_rate, early_stopping, max_attempts, seed,
        auc_train, auc_vld, acc_train, acc_vld, f1_train, f1_vld, train_time, nn_ga.fitness_curve[-1,0],
        file=open(output_file+'_results.txt', 'a'))

	return (nn_ga.fitness_curve)



def galrncrv(input_tuple):

	(pop_size, mutation_prob, max_iters, learning_rate, early_stopping, max_attempts, seed, 
     X_train, y_train, X_vld, y_vld, output_file, training_size) = input_tuple
	print(training_size, seed, file=open(output_file+'_progress.txt', 'a'))

	nn_ga = mlrose_hiive.NeuralNetwork(hidden_nodes = [4], activation = 'relu', \
	                                   algorithm = 'genetic_alg', max_iters = max_iters, \
	                                   bias = True, is_classifier = True, learning_rate = learning_rate, \
	                                   early_stopping = early_stopping, clip_max = 5.0, max_attempts = max_attempts, \
	                                   random_state = seed, pop_size=pop_size, mutation_prob=mutation_prob, curve=True)
	
	if training_size==1.0: 
		(X_train_red, y_train_red) = (X_train, y_train)
	else:
	# Shuffle data and then split into the training and the test set
		X_temp, X_train_red, y_temp, y_train_red = train_test_split(X_train, y_train, test_size = training_size, shuffle = True, random_state = 1988, stratify = y_train)

	tic = time.perf_counter()
	nn_ga.fit(X_train_red, y_train_red)
	toc = time.perf_counter()
	train_time = toc-tic

	# train
	y_train_pred = nn_ga.predict(X_train_red)
	auc_train = roc_auc_score(y_train_red, nn_ga.predicted_probs)
	acc_train = accuracy_score(y_train_red, y_train_pred)
	f1_train = f1_score(y_train_red, y_train_pred)
	# validation
	y_vld_pred = nn_ga.predict(X_vld)
	auc_vld = roc_auc_score(y_vld, nn_ga.predicted_probs)
	acc_vld = accuracy_score(y_vld, y_vld_pred)
	f1_vld = f1_score(y_vld, y_vld_pred)

	print(training_size, seed, auc_train, auc_vld, acc_train, acc_vld, f1_train, f1_vld, train_time, nn_ga.fitness_curve[-1,0],
    file=open(output_file+'_results.txt', 'a'))

	return (auc_train, auc_vld, acc_train, acc_vld, f1_train, f1_vld)




def worker2(x):
	print(x, file=open('output.txt', 'a'))
	return x*x