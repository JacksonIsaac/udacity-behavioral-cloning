## Import required packages
import csv
import cv2
import numpy as np

from scipy import ndimage
from sklearn.utils import shuffle

## List to store the rows of data in driving_log.csv
lines = []

## Input path for training data
input_path = '/opt/carnd_p3/data/'

with open(input_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    for line in reader:
        lines.append(line)

images, measurements = [], []

## Split the dataset for training and validation.
## Skip first row in lines, as it contains headers
from sklearn.model_selection import train_test_split
train_samples, valid_samples = train_test_split(lines[1:], shuffle=True, test_size=0.2)

import sklearn

## Generator function to read in rows of data,
## and return batches of data instead of whole data.
## This helps to split data into smaller data sets and decreases the
## load on memory and avoids memory related issues.
def generator(lines, batch_size=32):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        shuffle(lines)
        ## Loop over smaller parts of dataset.
        ## Each part will have data equal to the batch size.
        ## By default, 32 rows of data is computed
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]
            
            ## Store the image data and corresponding steering angle
            images, angles = [], []
            
            for batch_sample in batch_samples:
                filename = input_path + 'IMG/' + batch_sample[0].split('/')[-1]
                image = ndimage.imread(filename)
                measurement = float(batch_sample[3])

                images.append(image)
                angles.append(measurement)
                ## Since the train images consist of mostly left turns,
                ## to avoid overfitting to left turn,
                ## flip the image and change the sign of steering angle.
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(valid_samples, batch_size=batch_size)

## Dimensions of input image
ch, row, col = 3, 160, 320

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D

from math import ceil

model = Sequential()

## Normalize data
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape = (row, col, ch)))
## Trim image to only see section with road
model.add(Cropping2D(cropping=((70, 25), (0,0))))

## Create Model
## The model is inspired from the NVidia architecture
## as mentioned in Lecture 15. Even More Powerful Network
## Reference: https://devblogs.nvidia.com/deep-learning-self-driving-cars/

model.add(Conv2D(24,(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(48,(5,5),strides=(2,2),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))

model.add(Flatten())

model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(128))
model.add(Dense(84))
model.add(Dense(1))

## Print the model architecture summary
print(model.summary())

## Use mean square error and Adam optimizer.
model.compile(loss='mse', optimizer='Adam')

## When using generators, we need to use fit_generator() instead of fit()
model.fit_generator(train_generator, \
            steps_per_epoch=ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=ceil(len(train_samples)/batch_size), \
            epochs=5, verbose=1)

model.save('model_nvidia_arch_run3.h5')