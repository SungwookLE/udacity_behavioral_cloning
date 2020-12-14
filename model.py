# File: model.py
# Author: SungwookLE
# Date: '20.12/14 First Commit
# Note: The script used to create and train the model

import csv
import cv2
import numpy as np


lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements =[]
for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/'+ filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = measurement + correction
    steering_right = measurement - correction
    
    left_path = line[1]
    left_filename = left_path.split('/')[-1]
    current_path = 'data/IMG/'+ left_filename
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append(steering_left)
    
    
    right_path = line[2]
    right_filename = right_path.split('/')[-1]
    current_path = 'data/IMG/'+ right_filename
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append(steering_right)

lines = []
with open('wooks_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'wooks_data/IMG/'+ filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = measurement + correction
    steering_right = measurement - correction
    
    left_path = line[1]
    left_filename = left_path.split('/')[-1]
    current_path = 'wooks_data/IMG/'+ left_filename
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append(steering_left)
    
    
    right_path = line[2]
    right_filename = right_path.split('/')[-1]
    current_path = 'wooks_data/IMG/'+ right_filename
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append(steering_right)       


augmented_images, augmented_measurements =[], []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*(-1.0))
    
X_train = np.array(augmented_images)
Y_train = np.array(augmented_measurements)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, MaxPooling2D, Activation, Cropping2D

model = Sequential()


model.add(Lambda(lambda x: x / 255.0 -0.5 , input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
print('crop:\t{0}'.format(model.output_shape))

model.add( Conv2D(input_shape=(65,320,3), kernel_size=(5,5), filters=24, strides=(2,2), padding='SAME') )
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
model.fit(X_train, Y_train, validation_split=0.2, shuffle = True, nb_epoch=3, verbose=1)

#import matplotlib.pyplot as plt

# (12/14) performance good done
#model.save('model.h5')
model.summary()
