import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
training_set = train_datagen.flow_from_directory('Dataset_BreastCancer/training_set', target_size=(128,128), batch_size=32, class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('Dataset_BreastCancer/test_set', target_size=(128,128), batch_size=32, class_mode='categorical')

cnn = tf.keras.models.Sequential()
with strategy.scope():
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[128, 128, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=3, activation='softmax'))

    cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    cnn.fit(x=training_set, validation_data=test_set, epochs=49)


import numpy as np
from keras.preprocessing import image 
test_image = image.load_img('Dataset_BreastCancer/test_set/malignant/malignanttest.png', target_size = (128,128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0) 
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 0:
  prediction = 'b'
elif result[0][0]==1:
  prediction = 'm'
else:
  prediction = 'n'
print(prediction)