import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image 
import os
def train():
  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

  train_datagen = ImageDataGenerator(
      rescale=1./255,
      zoom_range=0.2,
      horizontal_flip=True,
      vertical_flip=True,
      rotation_range=20
  )
  global training_set
  training_set = train_datagen.flow_from_directory('Dataset_exter/training_set', target_size=(150,150), batch_size=32, class_mode='binary')

  test_datagen = ImageDataGenerator(rescale=1./255)
  test_set = test_datagen.flow_from_directory('Dataset_exter/test_set', target_size=(150,150), batch_size=32, class_mode='binary')
  global cnn
  cnn = tf.keras.models.Sequential()
  with strategy.scope():
      cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[150, 150, 3]))
      cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

      cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
      cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

      cnn.add(tf.keras.layers.Flatten())
      cnn.add(tf.keras.layers.Dense(units=150, activation='relu'))
      cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

      cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

      cnn.fit(x=training_set, validation_data=test_set, epochs=33)


def predict(filepath):
  test_image = image.load_img(filepath, target_size = (150,150))
  test_image = image.img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis=0) 
  result = cnn.predict(test_image)
  training_set.class_indices
  if result[0][0] == 0:
    prediction = 'n'
  else:
    prediction = 'y'
  return prediction