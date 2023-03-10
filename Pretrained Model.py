import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras=tf.keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Split the data
(raw_train,raw_validation,raw_test),metadata=tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]','train[80%:90%]','train[90%:]'],
    with_info=True,
    as_supervised=True)
print(type(raw_train))
# Create leabels
get_label_name = metadata.features['label'].int2str

# Convert all imges in the same size.
img_size=160

def format_example(image,label):
    image=tf.cast(image,tf.float32)
    image = image/255
    image = tf.image.resize(image,(img_size,img_size))
    return image,label
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# Picking pretrained model
IMH_SHAPE=(img_size,img_size,3)

base_model=tf.keras.applications.MobileNetV2(input_shape=IMH_SHAPE,
                                             include_top=False,
                                             weights='imagenet')


# Freezing the base
base_model.trainable = False

# Adding our classifier
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

# Training the Model
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
initial_epochs = 3
validation_steps=20
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)
acc = history.history['accuracy']
print(acc)
model.save("dogs_vs_cats.h5")
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')