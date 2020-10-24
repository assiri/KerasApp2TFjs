#https://deeplizard.com/learn/video/OO4HD-1wRN8
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt


mobile = tf.keras.applications.mobilenet.MobileNet()

# Lastly, we’re calling preprocess_input() from tf.keras.applications.mobilenet, which preprocesses the given image data to be in the same format as the images that MobileNet was originally trained on. Specifically, it’s scaling the pixel values in the image between -1 and 1, and this function will return the preprocessed image data as a numpy array.

def prepare_image(file):
    img_path = 'img/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

# from IPython.display import Image
# Image(filename='1.png', width=300,height=200) 

preprocessed_image = prepare_image('1.png')
predictions = mobile.predict(preprocessed_image)


# Then, we’re using an ImageNet utility function provided by Keras called decode_predictions(). It returns the top five ImageNet class predictions with the ImageNet class ID, the class label, and the probability. With this, we’ll be able to see the five ImageNet classes with the highest prediction probabilities from our model on this given image. Recall that there are 1000 total ImageNet classes.

results = imagenet_utils.decode_predictions(predictions)
print(results)

# MobileNet Espresso Prediction Let’s do another prediction, this time on this delicious looking cup of espresso.
#Image(filename='2.png', width=300,height=200)

preprocessed_image = prepare_image('2.png')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)

#MobileNet Strawberry Prediction
preprocessed_image = prepare_image('3.png')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)