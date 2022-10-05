# import datasets
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os

import pathlib
#dataset url using public dataset
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz" 
# #get the file from online
# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)  
# data_dir = pathlib.Path(data_dir)
# # full_dataset = datasets.ImageFolder(data_dir)

# image_count = len(list(data_dir.glob('*/*.jpg')))
# # print(image_count)
#! for dogs and cats
data_dir = '/home/yusuff/Desktop/Image_Class/dogs-vs-cats/train1'
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))

# roses = list(data_dir.glob('roses/*'))
# tulips = list(data_dir.glob('tulips/*'))

#*load the model
model = tf.keras.models.load_model("my_model_catsdogs.h5")
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)


class_names = train_ds.class_names
#* use a public image url to unit test the model

# flower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
flower_url = 'https://cdn.britannica.com/60/8160-050-08CCEABC/German-shepherd.jpg'
# flower_url = '/media/yusuff/56d31bd0-99ec-4644-853f-811982badc76/RX/temp/temp stuff/data/flower.jpeg'
sunflower_path = tf.keras.utils.get_file('German-shepherd', origin=flower_url)


img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch


predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)