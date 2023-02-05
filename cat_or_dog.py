import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale= 1./255, shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
training_set = train_datagen.flow_from_directory('training_set', target_size=(64,64), batch_size=32,class_mode='binary')
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('test_set', target_size=(64,64), batch_size=32, class_mode='binary')

# initialize neural network
cnn = tf.keras.models.Sequential()

# convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding Second Convolutional Layer.
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# flattening
cnn.add(tf.keras.layers.Flatten())

# Full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training CNN, Compiling CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# training cnn on training set and evaluating
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# make single prediction.
import numpy as np
from keras.preprocessing import image

test_image_1 = image.load_img('single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))

# convert a image into numpy array for prediction.
test_image_1 = image.img_to_array(test_image_1)

# adding extra dimension for batch
test_image_1 = np.expand_dims(test_image_1, axis=0)

result = cnn.predict(test_image_1/255.0)

# set cat/dog to integers.
training_set.class_indices

if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)



