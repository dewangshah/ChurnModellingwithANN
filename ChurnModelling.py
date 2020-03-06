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
from keras.layers import Dropout

#Initializing the ANN as a sequence of layers
classifier = Sequential()

#Adding the input and first hidden layer with dropout 
classifier.add(Dense(6,  kernel_initializer='uniform', activation='relu',input_dim=11))
classifier.add(Dropout(rate=0.1))#rate - a float between 0 and 1, it is a Fraction of input units to drop
#Adding the second hidden layer
classifier.add(Dense(6,  kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.1))
#Adding the output layer
classifier.add(Dense(1,  kernel_initializer='uniform', activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
"""
optimizer='adam' - Adam is one of the stocastic gradient descent optimizer algorithms
Metrics- Criterion to evaluate the model
"""
#Fitting the ANN to the Training set
classifier.fit(X_train,y_train,batch_size=10, nb_epoch=100)

#Predicting the Test Set Results
y_pred=classifier.predict(X_test)
y_pred=(y_pred > 0.5)

from sklearn.metrics import classification_report,confusion_matrix
report=classification_report(y_test,y_pred)
"""
              precision    recall  f1-score   support

           0       0.85      0.96      0.90      2378
           1       0.72      0.37      0.49       622

    accuracy                           0.84      3000
   macro avg       0.79      0.67      0.70      3000
weighted avg       0.83      0.84      0.82      3000
"""
matrix=confusion_matrix(y_test,y_pred)
"""
[[2289	89]
 [392	230]]

"""


"""
 Use our ANN model to predict if the customer with the following informations will leave the bank: 

    Geography: France
    Credit Score: 600
    Gender: Male
    Age: 40 years old
    Tenure: 3 years
    Balance: $60000
    Number of Products: 2
    Does this customer have a credit card ? Yes
    Is this customer an Active Member: Yes
    Estimated Salary: $50000

So should we say goodbye to that customer ?
"""
new_prediction = np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])
scaler.fit(new_prediction)
new_prediction=scaler.transform(new_prediction)
new_prediction=classifier.predict(new_prediction)
new_prediction= (new_prediction > 0.5)
#False