import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt 
import pickle
import numpy as np
import os
import cv2
import time

# load the image array and label array
X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0             # divide by 255 for normalizing our image array

# changed values in lists since  3 CNN layers each of size 64 with 0 dense layers was the most optimised out of 0,1,2 dense layers 1,2,3 CNN layers and 32,64,128 layer sizes
dense_layers = [0]      # number of dense layers
layer_sizes = [64]      # size of each layer
conv_layers = [3]       # number of CNN layers

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "BrainTumor-{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)
            tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))
            model = Sequential()
            
            model.add( Conv2D(layer_size, (3,3), input_shape = X.shape[1:]) )
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))
            
            for l in range(conv_layer-1):
                model.add( Conv2D(layer_size, (3,3)) )
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss="binary_crossentropy", 
                        optimizer='adam',
                        metrics=['accuracy'])

            model.fit(X, y, 
            batch_size=32,
            epochs=27, 
            validation_split=0.3, 
            callbacks=[tensorboard])

model.save('64x3-CNN-BrainTumor.model')  #saving the trained model
