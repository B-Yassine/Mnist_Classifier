import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
import keras
from keras.preprocessing.image import ImageDataGenerator

num_classes = 10
batch_size = 256
epochs = 64

mnist = tf.keras.datasets.mnist

# unpacks images to x_train/x_test and labels to y_train/y_test
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#you can print x_train[0] to visualize it
#print(x_train[0])

# scales data between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

from keras import backend as K

# input image dimensions
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Use the built in data generation features of Keras for "data augmentation"
datagen = ImageDataGenerator(
    width_shift_range = 0.075,
    height_shift_range = 0.075,
    rotation_range = 12,
    shear_range = 0.075,
    zoom_range = 0.05,
    fill_mode = 'constant',
    cval=0
)

datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

model.save('mnist_classifier.hdf5')
