import numpy as np
import glob
import cv2
import pandas as pd

classes = []
asso = {"Cofield" : 0, "Depuy" : 1, "Tornier" : 2, "Zimmer" : 3}
dim = (250, 250)
df = pd.DataFrame()
count = 0

files = glob.glob("*.jpg")
for img in files:
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, dim)
    resized_1D = resized.flatten()
    series = pd.Series(resized_1D)
    df = df.append(series, ignore_index=True)
    for key in asso.keys():
        if key in img:
            classes.append(asso[key])


df["labels"] = classes
print(df.describe(include="all"))
df.to_csv("project.csv", index=False)