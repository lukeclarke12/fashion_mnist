#!/usr/bin/env python
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that does verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# split the training and test data
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

class_names =  ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

trainX = trainX/255.0
testX = testX/255.0

#Here we are standizing all the pixel values between 0 - 1, by dividing by 255 which gives us the
#black and white image colorplt.figure(figsize=(10, 10))
#This is also a necessary preprocessing step we need to take to prepare the
#data before passing it to the neural network

#for i in range(25):
#    plt.subplot(5, 5, i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(trainX[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[trainy[i]])
#plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
# typical activation for output layer,notice 10 classes also in the output layer

model.compile(optimizer='adam', # how the model will be updated based on the data it recieves
              loss='sparse_categorical_crossentropy', # measures how well the model is learning
              metrics=['accuracy'])   

model.fit(trainX, trainy, epochs=5)


test_loss, test_acc = model.evaluate(testX, testy)


predictions = model.predict(testX)



print(predictions[0])
print("Predicted Label:", np.argmax(predictions[0]))
print("Actual Label:", testy[0])
print(f"Test Accuracy: {test_acc}")


