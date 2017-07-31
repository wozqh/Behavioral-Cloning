# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:11:13 2017

@author: Schaffer
"""

import csv
import cv2
import numpy as np

lines = []
with open('data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
#del(lines[0])
#####################################
from sklearn.model_selection import train_test_split
train_samples,validation_samples = train_test_split(lines,test_size=0.2)

import sklearn
import random

batch_size=32

def generator(samples,batch_size=32):
    num_samples = len(samples)
    random.shuffle(samples)
    while 1: #Loop forever so the generator never terminates
        #random.shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    if i==0:
                        correction = 0
                    elif i==1:
                        correction = 0.2
                    elif i==2:
                        correction = -0.2
                    source_path = line[i]
                    image = cv2.imread(source_path)
                    measurement = float(line[3])+correction
                    images.append(image)
                    measurements.append(measurement)
                    
                    images.append(cv2.flip(image,1))
                    measurements.append(-1.0*measurement)
              
#            aumented_images,aumented_measurements = [],[]
#            for image,measurement in zip(images,measurements):
#                aumented_images.append(image)
#                aumented_measurements.append(measurement)
#                aumented_images.append(cv2.flip(image,1))
#                aumented_measurements.append(measurement*-1.0)
                
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train,y_train)

#####################################

#images = []
#measurements = []
#
#for line in lines:
#    for i in range(3):
#        source_path = line[i]
#        current_path = source_path
#        #filename = source_path.split('/')[-1]
#        #current_path = 'data2/IMG/' + filename
#        if i == 0:
#            correction = 0
#        elif i == 1:
#            correction = 0.2
#        elif i == 2:
#            correction = -0.2
#        image = cv2.imread(current_path)
#        images.append(image)
#        measurement = float(line[3])+correction
#        measurements.append(measurement)
#    
#aumented_images,aumented_measurements = [],[]
#for image,measurement in zip(images,measurements):
#    aumented_images.append(image)
#    aumented_measurements.append(measurement)
#    aumented_images.append(cv2.flip(image,1))
#    aumented_measurements.append(measurement*-1.0)
#        
#X_train = np.array(aumented_images)
#y_train = np.array(aumented_measurements)

# compile and train the model using the generator function
train_generator = generator(train_samples,batch_size=32)
validation_generator = generator(validation_samples,batch_size=32)

#########################################
#for i in range(3):
#    x_batch,y_batch = next(train_generator)
#    print(x_batch.shape,y_batch.shape)
#########################################

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x:(x/255.0)-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')


#model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=7)
#history_object = model.fit(X_train,y_train,validation_split=0.2,shuffle=True,
#        nb_epoch=7,verbose=1)
#history_object = model.fit_generator(train_generator,samples_per_epoch=
#                                     len(train_samples),validation_data=validation_generator,
#                                     nb_val_samples=len(validation_samples),nb_epoch=7)
history_object = model.fit_generator(train_generator,samples_per_epoch=
                                     len(train_samples)/batch_size,validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples),nb_epoch=7)
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set','validation set'],loc='upper right')
plt.show()

model.save('model.h5')