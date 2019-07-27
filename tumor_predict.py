import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CATEGORIES = ["no","yes"] # categories of brain tumor
filepath = input("Enter image path: ") # input the image path

def prepare(filepath):
    # resizing and preprocessing image before prediction
    IMG_SIZE = 70   # initialize iamge size for image resizing
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read image in grayscale(since color images take hella lot of size) and store it in an array
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resize all the images in image_array by IMG_SIZExIMG_SIZE dimension and store it in an array
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) # return the reshaped image array when function is called

model = tf.keras.models.load_model("BrainTumor-CNN.model") # load the trained model for predicting 

prediction = model.predict([prepare(filepath)]) # predicting using the loaded model
img = mpimg.imread(filepath) # read image path for image display using matplotlib

if((CATEGORIES[int(prediction[0][0])])=="no"):
    print("No brain Tumor.")     # if prediction is 'no' then no brain tumor
elif((CATEGORIES[int(prediction[0][0])])=="yes"):	
    print("Yes brain tumor detected.")  # if prediction is 'yes' then brain tumor exists
else: 
    print("Error or abnormality or anamoly")  # error in prediction

# display the image
plt.imshow(img , cmap = plt.cm.binary)
plt.show()

