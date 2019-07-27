import cv2
import os
import tensorflow as tf 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

filepath = "testdata"
CATEGORIES = ["no","yes"]
IMG_SIZE = 70

model = tf.keras.models.load_model("64x3-CNN-BrainTumor.model")

for img in os.listdir(filepath):
    img_array = cv2.imread(os.path.join(filepath,img), cv2.IMREAD_GRAYSCALE) 
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    newarray = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    test = model.predict(newarray)

    img = mpimg.imread(os.path.join(filepath,img))

    if((CATEGORIES[int(test[0][0])])=="no"):
       print("No brain Tumor")
       plt.imshow(img , cmap = plt.cm.binary)
       plt.show()
    elif((CATEGORIES[int(test[0][0])])=="yes"):	
       print("Yes brain tumor detected.")
       plt.imshow(img , cmap = plt.cm.binary)
       plt.show()
    else:
       print("Error or abnormality or anamoly")    


    

