# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:36:35 2020

@author: Dewang
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:\\Users\\Dewang\\Desktop\\Practice Apps\\Deep Learning A-Z\\ANN\\Churn_Modelling.csv")

X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values


  
#Encoding the Categorical Data to Numerical Data 
#-----------
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_geography = LabelEncoder()
X[:,1]=labelencoder_geography.fit_transform(X[:,1])

labelencoder_gender = LabelEncoder()
X[:,2]=labelencoder_gender.fit_transform(X[:,2])

#Instead of 0 for France, 1 for Germany, 2 for Spain, we make two separate columns
oneHotEncoder=OneHotEncoder(categorical_features=[1])
X= oneHotEncoder.fit_transform(X).toarray()
X=X[:,1:]
#----------

#Feature Sacling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
X=scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6,  kernel_initializer='uniform', activation='relu',input_dim=11))
    classifier.add(Dense(6,  kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(1,  kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
"""
cv=no. of folds
n_jobs= no. of cpu's used, for running computations parallelly, since we are running the training 'cv' times,
each having 100 epochs
n_jobs = -1 - Use all CPUs
"""
accuracies = cross_val_score(estimator = classifier, X=X_train,y=y_train, cv=10, n_jobs= -1)

mean = accuracies.mean()
variance=accuracies.std()

