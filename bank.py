# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:11:12 2019

@author: Sushant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('bank-additional-full.csv',sep=";")

non_catagorical_values = ["age", "campaign", "pdays", "previous", "duration", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
categorical_values=["job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"]

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(dataset[categorical_values])
temp=encoder.transform(dataset[categorical_values]).toarray()

encoder_column=[str(i) for i in range(temp.shape[1])]
for value,i in enumerate(encoder_column):
    dataset[i]=temp[:,value]

dataset=dataset.drop(categorical_values,axis=1)

dataset["y"]=[1 if i=="yes" else 0 for i in dataset["y"]]
X = dataset.iloc[: , :-1].values  
y = dataset.iloc[: , -1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X ,y ,test_size = 0.3,random_state = 0 ,  stratify = y)

from sklearn.preprocessing import StandardScaler
sc_X =  StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train,Y_train)
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
print('Precision Score: ', precision_score(Y_test, y_pred))
print('Recall Score: ', recall_score(Y_test, y_pred))
print('f1 Score: ', f1_score(Y_test, y_pred))
print('accuracy Score: ', accuracy_score(Y_test, y_pred))

