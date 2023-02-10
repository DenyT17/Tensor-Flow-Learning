import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import keras.utils as imag
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Load data
(tr_in_images,tr_out_images),(ts_in_images,ts_out_images)=datasets.cifar10.load_data()
# Data Preprocessing
tr_in_images,ts_in_images=tr_in_images/255.0,ts_in_images/255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# creating a data generator object
datagen=ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

test_img=tr_in_images[15]
img=imag.img_to_array(test_img)
img=img.reshape((1,)+img.shape)

i=0

for batch in datagen.flow(img,save_prefix='test',save_format='jpeg'):
    plt.figure(i)
    plot=plt.imshow(imag.img_to_array(batch[0]))
    i+=1
    if i > 5:
        break
plt.show()
