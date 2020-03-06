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
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier(optimizer_name,hidden_layer_neuron_count):
    classifier = Sequential()
    classifier.add(Dense(hidden_layer_neuron_count,  kernel_initializer='uniform', activation='relu',input_dim=11))
    classifier.add(Dense(hidden_layer_neuron_count,  kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(1,  kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer_name,loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

#Dictionary containing the hyper parameters that we want to tune and optimize 
parameters = {'batch_size': [25,32],
             'epochs':[100,50,500],
             'optimizer_name':['adam','rmsprop'],
             'hidden_layer_neuron_count':[6,8,10]}

grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10)

grid_search = grid_search.fit(X_train,y_train)
best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_ 

"""
{'batch_size': 32, 'epochs': 500, 'hidden_layer_neuron_count': 8, 'optimizer_name': 'rmsprop'}

Best Accuracy: 0.8587142857142858
"""