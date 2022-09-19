"""
ConvolutionalNeuralNetworks.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
"""
from dataclasses import dataclass
from pickletools import optimize
from tabnanny import verbose
from tkinter.tix import InputOnly
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import time

import matplotlib.pyplot as plt

class ConvolutionalNeuralNetworks:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = models.Sequential()

    def ReshapeDataset(self):
        elemento = self.dataset.get_xtrain().shape[1]
        if elemento == 20:
            self.dataset.set_xtrain(self.dataset.get_xtrain().reshape(len(self.dataset.get_xtrain()), 5,4))
        else: 
            self.dataset.set_xtrain(self.dataset.get_xtrain().reshape(len(self.dataset.get_xtrain()), 6,4))
    
    def CreateModel(self):
        input_shape = self.dataset.get_xtrain().shape
        self.model = models.Sequential()
        self.model.add(layers.Conv2D( 64, (2,2), activation='relu', input_shape=(1, 5, 4) ))
        self.model.add(layers.Conv2D(128, (2,2), activation='relu'))
        self.model.add(layers.Conv2D(256, (2,2), activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))
        self.model.summary()
    
    def CompileModel(self):
        self.model.compile(optimizer='adam',
                            loss=tf.keras.losses.categorical_crossentropy,
                            metrics=['accuracy'])
        history = self.model.fit(   self.dataset.get_xtrain(),
                                    self.dataset.get_ytrain(),
                                    epochs=10, 
                                    verbose = 1,
                                    validation_data=(self.dataset.get_xvalidation(),self.dataset.get_yvalidation()))

    def ConvolutionalNeuralNetworks_run(self):
        self.ReshapeDataset()
        self.CreateModel()
        self.CompileModel()

class ConvolutionalNeuralNetworksMnist:
    def __init__(self):
        self.fig,self.ax = plt.subplots(1,1)
        self.history = None

        self.model = Sequential()
        self.batch_size = 128
        self.num_classes = 10
        self.epochs = 10

        # input image dimensions
        self.img_rows, self.img_cols = 28, 28

        # the data, split between train and test sets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        if K.image_data_format() == 'channels_first':
        #https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/ to know about image_data_format and what is "channelS_first"
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.img_rows, self.img_cols)
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 1)
            self.input_shape = (self.img_rows, self.img_cols, 1)
        
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255 #normalizing
        self.x_test /= 255 #normalizing
        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)
    
    def plt_dynamic(self, x, vy, ty, ax, colors=['b']):
        self.ax.plot(x, vy, 'b', label="Validation Loss")
        self.ax.plot(x, ty, 'r', label="Train Loss")
        plt.legend()
        plt.grid()
        self.fig.canvas.draw()

    def Model_Two_Capas(self):
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=self.input_shape))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.summary()
    
    def ModelCompile(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

        self.history = self.model.fit(self.x_train, self.y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=1,
                validation_data=(self.x_test, self.y_test))
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    
    def ModelEvaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        self.fig,self.ax = plt.subplots(1,1)
        self.ax.set_xlabel('epoch') ; self.ax.set_ylabel('Categorical Crossentropy Loss')
        # list of epoch numbers
        x = list(range(1,self.epochs+1))
        # print(history.history.keys())
        # dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
        # history = model_drop.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
        # we will get val_loss and val_acc only when you pass the paramter validation_data
        # val_loss : validation loss
        # val_acc : validation accuracy
        # loss : training loss
        # acc : train accuracy
        # for each key in histrory.histrory we will have a list of length equal to number of epochs
        vy = self.history.history['val_loss']
        ty = self.history.history['loss']
        self.plt_dynamic(x, vy, ty, self.ax)

    def ConvolutionalNeuralNetworksMnist_run(self):
        self.Model_Two_Capas()
        self.ModelCompile()
        self.ModelEvaluate()
