import cv2
import os
import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

filepath = "testdata"  # filepath for test data
CATEGORIES = ["no","yes"] # categories of brain tumor
IMG_SIZE = 70 # initialize image size for image resizing

model = tf.keras.models.load_model("BrainTumor-CNN.model") # load the trained model for testing 

for img in os.listdir(filepath):
    img_array = cv2.imread(os.path.join(filepath,img), cv2.IMREAD_GRAYSCALE) # read image in grayscale(since color images take hella lot of size) and store it in an array
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resize all the images in image_array by IMG_SIZExIMG_SIZE dimension and store it in an array
    newarray = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) # store the rshaped image array in a new array

    test = model.predict(newarray) # predicting using the loaded model

    img = mpimg.imread(os.path.join(filepath,img)) # read image path for image display using matplotlib
    
    # display the image when necessary using matplotlib

    if((CATEGORIES[int(test[0][0])])=="no"):
       print("No brain Tumor")    # if prediction is 'no' then no brain tumor
       plt.imshow(img , cmap = plt.cm.binary)
       plt.show()  
    elif((CATEGORIES[int(test[0][0])])=="yes"):	
       print("Yes brain tumor detected.") # if prediction is 'yes' then brain tumor is detected
       plt.imshow(img , cmap = plt.cm.binary)
       plt.show()
    else:
       print("Error or abnormality or anamoly")    # error in prediction


    

