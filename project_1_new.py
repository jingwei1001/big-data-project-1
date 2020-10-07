# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('new_data.csv') # read the training data

X=data.drop(["gameId" , "creationTime" , "winner"], axis = 1 ).values  # drop the meaningless data and the target label
y=data['winner'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # take twenty percent of all the data for test

kfold = model_selection.KFold(n_splits=10) # create a baagging classification with decision tree classificaton
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees)

results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean()) # the training error

model.fit(X_train,y_train) 
y_pred = model.predict(X_test) # test error using the data exacting from training data 
print("Accuracy:",accuracy_score(y_test, y_pred))


# read the test data and test the classification
test=pd.read_csv('test_set.csv') 
test_x=test.drop(["gameId" , "creationTime" , "winner"], axis = 1 ).values
test_y=test['winner'].values
pred_y=model.predict(test_x)

print("the basic information of classifier:\n",model)
print("Accuracy:",accuracy_score(test_y,pred_y))
#the classification is good, and can reach the accuracy of 0.968
