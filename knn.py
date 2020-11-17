from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, make_scorer, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, label_binarize
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("project.csv")

X = df.drop(["labels"], axis=1)
y = df["labels"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

my_scaler = MinMaxScaler()
X_train_scaled = my_scaler.fit_transform(X_train)
X_test_scaled = my_scaler.fit_transform(X_test)

my_knn = KNeighborsClassifier()

param_grid = {"n_neighbors" : [5, 10, 15, 20, 25, 30, 35, 40], "metric" : ["manhattan", "euclidean"]}
grid = GridSearchCV(estimator = my_knn, param_grid = param_grid, cv = 5, verbose=2)

grid.fit(X_train_scaled, y_train)

print("Best parameters =", grid.best_params_)

fitted_model = grid.best_estimator_

prediction = fitted_model.predict(X_test_scaled)

y_score = fitted_model.predict_proba(X_test_scaled)

print(accuracy_score(y_test, prediction))
print(recall_score(y_test, prediction, average="weighted"))
print(precision_score(y_test, prediction, average="weighted"))
print(f1_score(y_test, prediction, average="weighted"))

fpr = dict()
tpr = dict()
roc_auc = dict()
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]
colors = ["darkorange", "blue", "green", "red"]


for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='ROC curve class {} (area = {:.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curves')
plt.legend(loc="lower right")
plt.show()

