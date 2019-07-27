# image processing 
import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2
import random
import pickle

DATADIR = "E:\\m4xpl0it\\brain_tumor_detection\\Dataset" # define directory path
CATEGORIES = ["no","yes"]   # define sub-directories under a list which are going to be our categories

IMG_SIZE = 70  # since all images are not of same size we need to set a standard size for the image (or resize)

training_data = []  # create empty list for training data

def create_training_data():          # function to create training data
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # to traverse through images in the directory by joining with sub-directories
        class_num = CATEGORIES.index(category)  # to store the index of each image's category ,i.e, sub-directory
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) # read image in grayscale(since color images take hella lot of size) and store it in an array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resize all the images in image_array by IMG_SIZExIMG_SIZE dimension and store it in an array
                training_data.append([new_array, class_num]) # store the img_array and new_array in the training_data list
            except Exception as e:
                pass         # to continue our process even if few images in the dataset are corrupted

create_training_data() # calling the function to create the training data from our dataset

print(len(training_data))

random.shuffle(training_data) # shuffle training_data array to make the training of the two categories more effecient

X = []        # create empty list X
y = []        # create empty list y

for features,labels in training_data:
    X.append(features)                    # append new_array values to X list i.e the features
    y.append(labels)                      # append category or class_num to y list i.e the labels

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)    # reshaping values on X based on Image Size, gray scale and features

# to save the arrayed dataset and category for each so we can directly load it during training instead of running this everytime

pickle_out = open("X.pickle","wb")                    
pickle.dump(X, pickle_out)
pickle_out.close()        

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
