import os
import cv2
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve, auc


folder = 'data'

def create_labels(folder):
    labels = []
    for filename in os.listdir(folder):
        if "Cofield" in filename:
            labels.append(0)
        elif "Depuy" in filename:
            labels.append(1)
        elif "Tornier" in filename:
            labels.append(2)
        else:
            labels.append(3)
    return labels


# Loading the data
df = pd.read_csv("project.csv")
dataset = df.iloc[:,:-1].values
labels = create_labels(folder)
dataset = dataset.reshape(-1, 250, 250)

X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Convert into tensor the 4 arrays
X_train = Variable(torch.from_numpy(X_train).float())
X_test = Variable(torch.from_numpy(X_test).float())
y_train = Variable(torch.from_numpy(y_train).long())
y_test = Variable(torch.from_numpy(y_test).long())


# Make torch datasets from train and test sets
train = torch.utils.data.TensorDataset(X_train,y_train)
test = torch.utils.data.TensorDataset(X_test,y_test)

# Create train and test data loaders
train_loader = torch.utils.data.DataLoader(train, batch_size = 128, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, shuffle = True)



class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(250, 180)
        self.fc2 = nn.Linear(180, 150)
        self.fc3 = nn.Linear(150, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.1)

    # Feed Forward Function
    def forward(self, x):
        x = F.relu(self.fc1(x.float()))
        x = F.relu(self.fc2(x.float()))
        x = F.relu(self.fc3(x.float()))
        x = self.dropout(x.float())
        x = F.relu(self.fc4(x.float()))
        x = F.relu(self.fc5(x.float()))
        x = F.relu(self.fc6(x.float()))
        x = F.relu(self.fc7(x.float()))
        x = self.dropout(x.float())
        x = self.output_layer(x.float())
        
        # Return the created model
        return x


#######################################
model = ANN()
criterion = nn.CrossEntropyLoss()   # cross entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(200):
    optimizer.zero_grad()
    out = model(X_train)
    y_train = y_train.view(-1,1)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print ('Number of epoch :', epoch, '- Loss :', loss.item())

predict_out = model(X_test)
_, predict_y = torch.max(predict_out, 1)

print('Accuracy :', accuracy_score(y_test.data, predict_y.data)*100)
print('F1 :', f1_score(y_test.data, predict_y.data, average='weighted'))
print('Recall :', recall_score(y_test.data, predict_y.data, average="weighted", zero_division=1))
print("Precision :", precision_score(y_test.data, predict_y.data, average="weighted"))



### lr = 0.05, batch = 64, iter = 50, acc = 4.2, F1 = 0.04, recall = 0.09, precision = 0.088
### lr = 0.05, batch = 64, iter = 100, acc = 17.5, F1 = 0.13, recall = 0.19, precision = 0.244
### lr = 0.05, batch = 64, iter = 200, acc = 20.8, F1 = 0.15, recall = 0.25, precision = 0.175

### lr = 0.05, batch = 128, iter = 50, acc = 5.0, F1 = 0.07, recall = 0.05, precision = 0.135
### lr = 0.05, batch = 128, iter = 100, acc = 14.7, F1 = 0.09, recall = 0.12, precision = 0.148
### lr = 0.05, batch = 128, iter = 200, acc = 19.8, F1 = 0.13, recall = 0.22, precision = 0.201