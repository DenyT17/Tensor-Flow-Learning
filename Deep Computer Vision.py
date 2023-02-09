import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load data
(tr_in_images,tr_out_images),(ts_in_images,ts_out_images)=datasets.cifar10.load_data()
# Data Preprocessing
tr_in_images,ts_in_images=tr_in_images/255.0,ts_in_images/255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# img_index=25
#
# plt.imshow(tr_in_images[img_index],cmap=plt.cm.binary)
# plt.xlabel(class_names[tr_out_images[img_index][0]])
# plt.show()

# Building Convolutions Base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Adding Dense Layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Training
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(tr_in_images, tr_out_images, epochs=4,
                    validation_data=(ts_in_images, ts_out_images))

# Evaluating the Model

test_loss,test_acc=model.evaluate(ts_in_images,
                                  ts_out_images,
                                  verbose=2)
print(test_acc)