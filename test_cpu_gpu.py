# Script para evaluar el tiempo de procesamiento enc omputadoras con gpus y cpus
# Original scripts
# https://www.analyticsvidhya.com/blog/2021/11/benchmarking-cpu-and-gpu-performance-with-tensorflow/


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from time import time


##################
# TEST 1 digits  #
##################

# Separate train and test dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Scaling the image, transform data from 8bit values to 0-1 range
X_train_scaled = X_train/255
X_test_scaled = X_test/255
# one hot encoding labels
y_train_encoded = keras.utils.to_categorical(y_train, num_classes = 10, dtype = 'float32')
y_test_encoded = keras.utils.to_categorical(y_test, num_classes = 10, dtype = 'float32')

def get_model():
    """This function is for convenience. We can create exctly the same model easly changing
the device we are going to use """
    model = keras.Sequential([
        # get a flat representation of images (NO convolution here)
        keras.layers.Flatten(input_shape=(32,32,3)),
        # Creatinty 'almost' one neuron by pixel (3072)
        keras.layers.Dense(3000, activation='relu'),
        # reducing dimentions
        keras.layers.Dense(1000, activation='relu'),
        # multiclass classification
        keras.layers.Dense(10, activation='sigmoid')    
    ])
    # model creation-compilation and loss funciton selection
    model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


# CPU
# using specific device
t1 = time()
with tf.device('/CPU:0'):
    model_cpu = get_model()
    model_cpu.fit(X_train_scaled, y_train_encoded, epochs = 10)
t2 = time()
test1_cpu_time = t2-t1
    
# GPU
t1 = time()
with tf.device('/GPU:0'):
    model_gpu = get_model()
    model_gpu.fit(X_train_scaled, y_train_encoded, epochs = 10)
t2 = time()
test1_gpu_time = t2-t1


###################
# TEST 2 fashion  #
###################

# loading dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# checking images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# scaling, from 8bit pixels to 0-1 values
train_images_scaled = train_images / 255.0
test_images_scaled = test_images / 255.0


def get_model(hidden_layers=1):
    # Flatten layer for input
    """This function is for convenience. We can create exctly the same model easly changing
the device we are going to use """
    # Create list of layers, first one is a flatten layer
    layers = [keras.layers.Flatten(input_shape=(28, 28))]
    # hideen layers, how many?
    for i in range(hidden_layers):
        layers.append(keras.layers.Dense(500, activation='relu'),)
    # output layer, classification
    layers.append(keras.layers.Dense(10, activation='sigmoid'))
    # creatin model 
    model = keras.Sequential(layers)
    # compiling and selection optimizer and loss funcition
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


t1 = time()
with tf.device('/CPU:0'):
    # use 5 layers
    cpu_model = get_model(hidden_layers=5)
    cpu_model.fit(train_images_scaled, train_labels, epochs=5)
t2 = time()
test2_cpu_time = t2-t1

    
t1 = time()
with tf.device('/GPU:0'):
    gpu_model = get_model(hidden_layers=5)
    gpu_model.fit(train_images_scaled, train_labels, epochs=5)
t2 = time()
test2_gpu_time = t2-t1


##########
# TEST 3 #
##########
# what can we add here?, something dificult



##########
# Report #
##########

print('Speed test:')
print(f'Test 1 CPU:  {test1_cpu_time} seconds')
print(f'Test 1 GPU:  {test1_gpu_time} seconds')
print(f'Test 2 CPU:  {test2_cpu_time} seconds')
print(f'Test 2 CPU:  {test2_gpu_time} seconds')

