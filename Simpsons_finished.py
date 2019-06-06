# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:35:02 2018

@author: Matthew
"""
import numpy as np
import pandas as pd
import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications import VGG16


def plotHist(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()

base_dir = 'E:\\py599\\the-simpsons-dataset\\'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Directory with our training Homer pictures
train_Homer_dir = os.path.join(train_dir, 'Homer')
train_Marge_dir = os.path.join(train_dir, 'Marge')
train_Lisa_dir = os.path.join(train_dir, 'Lisa')
train_Bart_dir = os.path.join(train_dir, 'Bart')

# Directory with our validation Homer pictures
validation_Homer_dir = os.path.join(validation_dir, 'Homer')
validation_Marge_dir = os.path.join(validation_dir, 'Marge')
validation_Lisa_dir = os.path.join(validation_dir, 'Lisa')
validation_Bart_dir = os.path.join(validation_dir, 'Bart')

# Directory with our test Homer pictures
test_Homer_dir = os.path.join(test_dir, 'Homer')
test_Marge_dir = os.path.join(test_dir, 'Marge')
test_Lisa_dir = os.path.join(test_dir, 'Lisa')
test_Bart_dir = os.path.join(test_dir, 'Bart')
#%%
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        color_mode='rgb',
        shuffle=True,
        seed=50,
        batch_size=20,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        color_mode='rgb',
        shuffle=True,
        seed=50,
        batch_size=20,
        class_mode='categorical')


test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        color_mode='rgb',
        shuffle=True,
        seed=50,
        batch_size=20,
        class_mode='categorical')

stepTrain = train_generator.n//train_generator.batch_size
stepValid = validation_generator.n//validation_generator.batch_size
stepTest = test_generator.n//test_generator.batch_size

history = model.fit_generator(generator=train_generator,
                               steps_per_epoch=stepTrain,
                               validation_data=validation_generator,
                               validation_steps=stepValid,
                               epochs=50,
                               verbose = 0)

plotHist(history)
test_loss, test_acc = model.evaluate_generator(test_generator, steps=stepTest)
print('test acc:', test_acc)

#%%

conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150, 150, 3))
model2 = models.Sequential()
model2.add(conv_base)
model2.add(layers.Flatten())
model2.add(layers.Dense(256, activation='relu'))
model2.add(layers.Dense(4, activation='softmax'))

conv_base.trainable = False
preTrain_datagen = ImageDataGenerator(rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
preTrainTest_datagen = ImageDataGenerator(rescale=1./255)
preTrain_generator = preTrain_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        color_mode='rgb',
        shuffle=True,
        seed=50,
        batch_size=20,
        class_mode='categorical')
preValidation_generator = preTrainTest_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        color_mode='rgb',
        shuffle=True,
        seed=50,
        batch_size=20,
        class_mode='categorical')

model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

#optimizers.RMSprop(lr=2e-5)
stepTrain2 = preTrain_generator.n//preTrain_generator.batch_size
stepValid2 = preValidation_generator.n//preValidation_generator.batch_size
history2 = model2.fit_generator(
      preTrain_generator,
      steps_per_epoch=stepTrain2,
      epochs=50,
      validation_data=preValidation_generator,
      validation_steps=stepValid2,
      verbose=0)

plotHist(history2)

preTest_generator = preTrainTest_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        color_mode = 'rgb',
        shuffle=True,
        seed=50,
        class_mode='categorical')

test_loss2, test_acc2 = model2.evaluate_generator(preTest_generator, steps=50)
print('test acc:', test_acc2)
