# File: model.py
# Author: SungwookLE
# Date: '20.12/14 First Commit
# Note: The script used to create and train the model

import csv
import cv2
import numpy as np
import math
import glob

log_files = glob.glob('./data/*.csv')

# (12/15) preprocess for coroutine batch
samples=[]

for log_file in log_files:
    print(log_file)
    with open(log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

        del samples[0]

import os
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn

def generator(samples, batch_size=32):
        num_samples = len(samples)

        while 1: #Loop forever so the generator never terminates
                sklearn.utils.shuffle(samples) #test what happen if this line comment out
                # create adjusted steering measurements for the side camera images
                correction = 0.32 # this is a parameter to tune

                for offset in range(0, num_samples, batch_size):
                        batch_samples = samples[offset: offset+batch_size]
                        images = []
                        measurements = []
                        
                        for batch_sample in batch_samples:
                            name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                            center_image = cv2.imread(name)
                            center_angle = float(batch_sample[3])
                            images.append(center_image)
                            measurements.append(center_angle)
                            #flip of center cam
                            images.append(cv2.flip(center_image,1))
                            measurements.append(center_angle*(-1.0))

                            steering_left = center_angle + correction
                            left_name = 'data/IMG/'+batch_sample[1].split('/')[-1]
                            left_image = cv2.imread(left_name)
                            images.append(left_image)
                            measurements.append(steering_left)
                            #flip of left cam
                            images.append(cv2.flip(left_image,1))
                            measurements.append(steering_left*(-1.0))

                            steering_right = center_angle - correction
                            right_name = 'data/IMG/'+batch_sample[2].split('/')[-1]
                            right_image = cv2.imread(right_name)
                            images.append(right_image)
                            measurements.append(steering_right)
                            #flip of left cam
                            images.append(cv2.flip(right_image,1))
                            measurements.append(steering_right*(-1.0))
                            
                        X_train = np.array(images)
                        Y_train = np.array(measurements)

                        X_train = np.resize(X_train, (batch_size, 160, 320, 3))
                        Y_train = np.resize(Y_train, (batch_size, 1))

                        yield sklearn.utils.shuffle(X_train, Y_train)
               
# Set out batch size
batch_size = 32

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = batch_size)
validation_generator = generator(validation_samples, batch_size = batch_size)

# Learning Architecture
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, MaxPooling2D, Activation, Cropping2D

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 -0.5 , input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
print('crop:\t{0}'.format(model.output_shape))

model.add(Conv2D(input_shape=(65,320,3), kernel_size=(5,5), filters=24, strides=(2,2), padding='SAME') )
model.add(Dropout(rate=0.5))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='SAME'))
model.add(Activation('relu'))    
print('conv2d 1:\t{0}'.format(model.output_shape))

model.add( Conv2D(input_shape=(33,160,24), kernel_size=(5,5), filters=36, strides=(2,2), padding='SAME') )
model.add(Dropout(rate=0.5))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='SAME'))
model.add(Activation('relu')) 
print('conv2d 2:\t{0}'.format(model.output_shape))

model.add( Conv2D(input_shape=(17,80,36), kernel_size=(5,5), filters=48, strides=(2,2), padding='SAME') )
model.add(Dropout(rate=0.5))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='SAME'))
model.add(Activation('relu')) 
print('conv2d 3:\t{0}'.format(model.output_shape))

model.add( Conv2D(input_shape=(9,40,48), kernel_size=(3,3), filters=64, strides=(1,1), padding='SAME') )
model.add(Dropout(rate=0.5))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='SAME'))
model.add(Activation('relu')) 
print('conv2d 4:\t{0}'.format(model.output_shape))

model.add( Conv2D(input_shape=(9,40,64), kernel_size=(3,3), filters=64, strides=(1,1), padding='SAME') )
model.add(Dropout(rate=0.5))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='SAME'))
model.add(Activation('relu')) 
print('conv2d 5:\t{0}'.format(model.output_shape))

model.add(Flatten(input_shape=(90,40,64)))
print('flatten:\t{0}'.format(model.output_shape))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#(12/14) whole batch process
#model.fit(X_train, Y_train, validation_split=0.2, shuffle = True, nb_epoch=3, verbose=1)

#(12/16) batch using generator(coroutine)
history_object= model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)*6/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)*6/batch_size), 
            epochs=5, verbose=1)

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

# (12/14) 
model.save('model.h6')
model.summary()

plt.savefig('learning_epochs.png')
plt.show()
