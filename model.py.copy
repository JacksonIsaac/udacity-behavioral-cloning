import csv
import cv2
import numpy as np

from scipy import ndimage
from sklearn.utils import shuffle

lines = []

input_path = '/opt/carnd_p3/data/'

with open(input_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    for line in reader:
        lines.append(line)
        
images, measurements = [], []
aug_images, aug_measurements = [], []

# for line in lines[1:]:
#     source_path = line[0]
#     filename = source_path.split('/')[-1]
#     current_path = input_path + 'IMG/' + filename
    
#     ## Read the image as RGB.
#     image = ndimage.imread(current_path)
#     images.append(image)
    
#     measurements.append(float(line[3]))
    
# for image, measurement in zip(images, measurements):
#     aug_images.append(image)
#     aug_measurements.append(measurement)
#     aug_images.append(cv2.flip(image,1))
#     aug_measurements.append(measurement*-1.0)

# X_train, y_train = np.array(images), np.array(measurements)
# X_train, y_train = shuffle(X_train, y_train)

from sklearn.model_selection import train_test_split
# X_train, y_train, X_valid, y_valid = train_test_split(X_train, y_train, test_size=0.2)

import sklearn

# def generator(X_train, y_train, batch_size=32):
#     num_samples = len(y_train)
#     while 1: # Loop forever so the generator never terminates
# #         shuffle(X_train, y_train)
#         for offset in range(0, num_samples, batch_size):
            
#             images = []
#             angles = []
#             for image, measurement in zip(X_train[offset:offset+batch_size], y_train[offset:offset+batch_size]):
#                 images.append(image)
#                 angles.append(measurement)
#                 images.append(cv2.flip(image,1))
#                 angles.append(measurement*-1.0)

#             # trim image to only see section with road
#             X_train = np.array(images)
#             y_train = np.array(angles)
#             yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(X_train, y_train, batch_size=batch_size)
validation_generator = generator(X_valid, y_valid, batch_size=batch_size)

ch, row, col = 3, 160, 320

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D

from math import ceil

model = Sequential()

## Normalize data
# model.add(Lambda(lambda x: x/127.5 - 1.,
#         input_shape=(row, col, ch),
#         output_shape=(row, col, ch)))
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape = (row, col, ch)))
model.add(Cropping2D(cropping=((70, 25), (0,0))))

## Create Model
model.add(Conv2D(6,(5,5),strides=(2,2),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(6,(5,5),strides=(2,2),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))


# model.add(Conv2D(24,(5,5),strides=(2,2),activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(36,(5,5),strides=(2,2),activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(48,(5,5),strides=(2,2),activation='relu'))
# # model.add(MaxPooling2D())
# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(Flatten())
# # model.add(Flatten())
# # model.add(Dense(2112))
# # model.add(Dense(100))
# # model.add(Dense(50))
# # model.add(Dense(10))
# # model.add(Dense(128))
# # model.add(Dense(84))
# # model.add(Dense(1))

print(model.summary())

model.compile(loss='mse', optimizer='Adam')
# model.fit(X_train, y_train, validation_split=0.25, shuffle=True, epochs=15)
model.fit_generator(train_generator, \
            steps_per_epoch=ceil(len(y_train)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=ceil(len(y_valid)/batch_size), \
            epochs=5, verbose=1)

model.save('model_nvidia_arch_run1.h5')