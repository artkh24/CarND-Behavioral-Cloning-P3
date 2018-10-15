import sklearn
import cv2
import csv
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, Activation
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def data_preprocessing(datapaths, correction=0.2):
    car_images = []
    steering_angles = []
    for datapath in datapaths:
        with open(datapath+'/driving_log.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            for row in reader:
                steering_center = float(row[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                
                img_center = row[0].strip()
                img_left =  row[1].strip()
                img_right = row[2].strip()
                
                # add images and angles to data set
                car_images.extend([img_center, img_left, img_right])
                steering_angles.extend([steering_center, steering_left, steering_right])
    return car_images, steering_angles


# Generating images and measurements
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        samples=sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                center_angle = float(batch_sample[1])
                images.append(center_image)
                angles.append(center_angle)
                
                #data augmentation
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)


def nVidiaNet():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(((65, 25), (0, 0)), input_shape=(3, 160, 320)))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2)))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


paths=['/home/workspace/CarND-Behavioral-Cloning-P3/track1',
      '/home/workspace/CarND-Behavioral-Cloning-P3/track2']
# preprocess images
images, measurements = data_preprocessing(paths, correction=0.2)
print('Image Count ' + str(len(images)))

# spliting samples to train and validation sets
train_samples, validation_samples = train_test_split(list(zip(images, measurements)), test_size=0.2)
print('Train Count ' + str(len(train_samples)))
print('Validation Count ' + str(len(validation_samples)))

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Neural Net
model = nVidiaNet()

# Training the model
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,\
                                     nb_val_samples=len(validation_samples),\
                                     nb_epoch=5, verbose=1)
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

