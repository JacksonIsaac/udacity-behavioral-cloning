import csv
import cv2
import numpy as np

from scipy import ndimage

lines = []

input_path = '/opt/carnd_p3/data/'

with open(input_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    
    for line in reader:
        lines.append(line)
        
images, measurements = [], []
aug_images, aug_measurements = [], []

for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = input_path + 'IMG/' + filename
    
    ## Read the image as RGB.
    image = ndimage.imread(current_path)
    images.append(image)
    
    measurements.append(float(line[3]))
    
for image, measurement in zip(images, measurements):
    aug_images.append(image)
    aug_measurements.append(measurement)
    aug_images.append(cv2.flip(image,1))
    aug_measurements.append(measurement*-1.0)
    
X_train, y_train = np.array(aug_images), np.array(aug_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D

model = Sequential()

## Normalize data
model.add(Cropping2D(cropping=((90, 25), (0,0)), input_shape = (160,320,3)))
model.add(Lambda(lambda x: (x/255.0) - 0.5))

## Create Model
model.add(Conv2D(24,5,5,activation='relu'))
# model.add(MaxPooling2D())
model.add(Conv2D(36,5,5,activation='relu'))
# model.add(MaxPooling2D())
model.add(Conv2D(48,3,3,activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
# model.add(Dense(128))
# model.add(Dense(84))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='Adam')
model.fit(X_train, y_train, validation_split=0.25, shuffle=True, epochs=15)

model.save('model_nvidia_arch_run1.h5')