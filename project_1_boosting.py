import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# read the data for training
data = pd.read_csv('new_data.csv')
#extract meaningful attributes
X=data.drop(["gameId" , "creationTime" , "winner"], axis = 1 ).values
y=data['winner'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

num_trees = 10
kfold = model_selection.KFold(n_splits=10)
# create a boosting classification
model = AdaBoostClassifier(n_estimators=num_trees)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))


#read the test data and test the classification accuracy
test=pd.read_csv('test_set.csv')
test_x=test.drop(["gameId" , "creationTime" , "winner"], axis = 1 ).values
test_y=test['winner'].values
pred_y=model.predict(test_x)
print("the basic informaton of classifier",model)
print("Accuracy:",accuracy_score(test_y,pred_y ))





