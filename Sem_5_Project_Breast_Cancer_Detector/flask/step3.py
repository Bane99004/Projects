import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image 
def train_2():
  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

  train_datagen = ImageDataGenerator(
      rescale=1./255,
      shear_range=0.1,
      zoom_range=0.2,
      horizontal_flip=True,
      vertical_flip=True,
      brightness_range=[0.8, 1.2],
      width_shift_range=0.1,
      height_shift_range=0.1,
      rotation_range=20
  )
  global training_set

  training_set = train_datagen.flow_from_directory('Dataset_BreastCancer/training_set', target_size=(150,150), batch_size=32, class_mode='categorical')

  test_datagen = ImageDataGenerator(rescale=1./255)
  test_set = test_datagen.flow_from_directory('Dataset_BreastCancer/test_set', target_size=(150,150), batch_size=32, class_mode='categorical')
  global cnn
  
  cnn = tf.keras.models.Sequential()
  with strategy.scope():
      cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[150, 150, 3]))
      cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

      cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
      cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
      cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
      cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
      cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
      cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
      cnn.add(tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu'))
      cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
      # cnn.add(tf.keras.layers.Conv2D(filters=1024, kernel_size=3, activation='relu'))
      # cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

      cnn.add(tf.keras.layers.Flatten())
      cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
      cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

      cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

      cnn.fit(x=training_set, validation_data=test_set, epochs=56)
      # predictions = cnn.predict(test_set)

    
      # predicted_labels = np.argmax(predictions, axis=1)
  
     
      # true_labels = test_set.classes
      # print(predicted_labels)
      # print("True")
      # print(true_labels)
      # print(np.concatenate((predicted_labels.reshape(len(predicted_labels),1), true_labels.reshape(len(true_labels),1)),1))
# train_2()

def predict_2(filepath):
    
    test_image = image.load_img(filepath, target_size=(128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0) 
    test_image = test_image / 255.0 
    result = cnn.predict(test_image)
    predicted_index = np.argmax(result, axis=1)[0]
    class_labels = {v: k for k, v in training_set.class_indices.items()}
    prediction = class_labels[predicted_index]

    return prediction


