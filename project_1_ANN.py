# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# read the training data
data = pd.read_csv('E:/new_data.csv')

#create an SimpleImputer
SimpleImputer = SimpleImputer()

# extract the attributes and change the range of each attribute to [0,1] 
X = data.drop(['gameId','creationTime','winner'], axis=1).values
X = SimpleImputer.fit_transform(X)
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# extract the label
y = data['winner'].values
# change the attibutes and labels to float type
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
# define an ANN classifier
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=18, out_features=72)
        self.output = nn.Linear(in_features=72, out_features=3)
        nn.Softmax(dim=1) 
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.output(x)
        x = F.softmax(x)
        return x
    
model = ANN()
# difine the loss and learning rate
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# learning procedure, the data used for learning is 2000
epochs = 2000
loss_arr = []
for i in range(epochs):
    y_hat = model.forward(X_train)
    loss = criterion(y_hat, y_train)
    loss_arr.append(loss)
    if i % 10 == 0:
        print(f'Epoch: {i} Loss: {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# predict the label of each x_test 
predict_out = model(X_test)
_,predict_y = torch.max(predict_out, 1)
# calculate the accuracy rate using data from training data
from sklearn.metrics import accuracy_score
print("The accuracy is ", accuracy_score(y_test, predict_y) )


# read and process the data used for test
test=pd.read_csv('test_set.csv')
test_x = test.drop(["gameId" , "creationTime" , "winner"], axis = 1 ).values
test_x = SimpleImputer.fit_transform(test_x)
scaler = MinMaxScaler(feature_range=(0, 1))
test_x = scaler.fit_transform(test_x)
test_x = torch.FloatTensor(test_x)

# use the ANN classifier to predict label
pred_out = model(test_x)
_,pred_y = torch.max(pred_out, 1)

test_y = test['winner'].values
test_y = torch.FloatTensor(test_y)
# the final accuracy rate is 0.967
print("the basic information of classifier:\n",model)
print("Accuracy:",accuracy_score(test_y,pred_y ))


