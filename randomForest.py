#! /usr/bin/python3

import numpy as np
import glob
import cv2
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize, label_binarize
from sklearn import metrics

from skimage.transform import rotate
from skimage import io

import time
import os
import sys
import random
import shutil
import matplotlib.pyplot as plt


def run_random_forest():
    data = pd.read_csv("project.csv")

    df_x = data.iloc[1:, :-1]
    df_y = data.iloc[1:, -1]

    df_x_normalized = normalize(df_x)
    # print(df_x_normalized)


    x_train, x_test, y_train, y_test = train_test_split(df_x_normalized, df_y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier()
    # bootstrap=False mauvaise idée car réduit l'acuracy
    # random_state=4 => accuracy : 0.54

    param_grid = {"n_estimators" : [100,200,300,400,500], "criterion" : ["gini", "entropy"], "bootstrap" :["True", "False"], "random_state" : [4], "n_jobs" : [10]}
    grid = GridSearchCV(estimator = rf, param_grid=param_grid, cv = 5 , verbose=2)


    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_

    predict = best_model.predict(x_test)

    # For the ROC curve
    y_score = best_model.predict_proba(x_test)

    # Calculate Metrics scores
    accuracy = metrics.accuracy_score(y_test.values, predict)
    precision = metrics.precision_score(y_test.values, predict, labels=[0, 1, 2, 3], average='weighted')
    recall = metrics.recall_score(y_test.values, predict, labels=[0, 1, 2, 3], average='weighted')
    f1_score = metrics.f1_score(y_test.values, predict, labels=[0, 1, 2, 3], average='weighted')

    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    n_classes = y_test_bin.shape[1]    
    colors = ["darkorange", "blue", "green", "red"]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='ROC curve class {} (area = {:.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves')
    plt.legend(loc="lower right")
    plt.show()


    print("Accuracy :\t{}\nPrecision :\t{}\nRecall :\t{}\nF1_Score :\t{}\n".format(accuracy, precision, recall, f1_score))



if __name__ == "__main__":
    start_time = time.time()
    args = sys.argv
    if len(args) > 1:
        print(type(args[1]))
        # exit()
        if args[1] == '1':
            data_augmentation()
            img_to_csv()
        elif args[1] == '2':
            img_to_csv()
    run_random_forest()
    print("Done in : {}".format(time.time()-start_time))



# HELP !!!


# Les images doivent être dans le dossier data, situé dans le meme directory que le sript.
# Au run, il va créé le csv dans le même directory.

# Plusieurs run possibles :
# 1. - Data augmentation + creation du CSV + prediction
# 2.  - Creation du CSV + prediction
# 3.  - Prediction


# Pour le 1, executer :     ./randomForest.py 1

# Pour le 2, executer :     ./randomForest.py 2

# Pour le 3, executer :     ./randomForest.py

