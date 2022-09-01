"""
ConvolutionalNeuralNetworks.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
"""
from dataclasses import dataclass
from pickletools import optimize
from tkinter.tix import InputOnly
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense
class ConvolutionalNeuralNetworks:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = models.Sequential()

    def ReshapeDataset(self):
        self.dataset.set_xtrain(self.dataset.get_xtrain().reshape(len(self.dataset.get_xtrain()), 6,4))
    
    def CreateModel(self):
        input_shape = self.dataset.get_xtrain().shape
        self.model = models.Sequential()
        self.model.add(layers.Conv2D( 64, (2,2), activation='relu', input_shape=(5, 7, 1) ))
        self.model.add(layers.Conv2D(128, (2,2), activation='relu'))
        self.model.add(layers.Conv2D(256, (2,2), activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))
        self.model.summary()
    
    def CompileModel(self):
        self.model.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])
        history = self.model.fit(self.dataset.get_xtrain(),self.dataset.get_ytrain(), epochs=10, validation_data=(self.dataset.get_xvalidation(),self.dataset.get_yvalidation()))

    def ConvolutionalNeuralNetworks_run(self):
        self.ReshapeDataset()
        self.CreateModel()
        self.CompileModel()
    
