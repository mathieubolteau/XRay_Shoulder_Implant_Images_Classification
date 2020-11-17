
import glob
import cv2
import os
import random
import shutil

def data_augmentation():
    print("DATA AUGMENTATION ...")
    target = "./augmented_data/"
    if os.path.isdir(target):
        shutil.rmtree(target)
    os.mkdir("./augmented_data/")
    
    for file in glob.glob("data/*.jpg"):
        shutil.copy(file, target)
        base = os.path.basename(file)
        name_file = os.path.splitext(base)
        image = io.imread(file)
        rotated = rotate(image, random.choice([15, -15]))
        resized = cv2.resize(rotated, (250,250))
        io.imsave(target+name_file[0]+"_1"+name_file[1], resized)

