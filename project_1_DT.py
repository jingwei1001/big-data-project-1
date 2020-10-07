# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split

# read the training data
data = pd.read_csv('new_data.csv')

X=data.drop(["gameId" , "creationTime" , "winner"], axis = 1 ).values
y=data['winner'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#create a decision classifier
clf = DecisionTreeClassifier()
clf=clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)# test error using the data exacting from training data
print("Accuracy:",accuracy_score(y_test, y_pred))


# read the test data and test the classification accuracy
test=pd.read_csv('test_set.csv')
test_x=test.drop(["gameId" , "creationTime" , "winner"], axis = 1 ).values
test_y=test['winner'].values
pred_y=clf.predict(test_x)
print("the basic information of classifier:\n",clf)
print("Accuracy:",accuracy_score(test_y,pred_y ))
