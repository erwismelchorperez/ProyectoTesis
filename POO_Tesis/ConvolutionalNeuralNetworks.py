"""
ConvolutionalNeuralNetworks.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
"""
from dataclasses import dataclass
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
        x_aux = []
        for x in self.dataset.get_xtrain():
            x = np.reshape(x,(4,6))
            x_aux.append(x)
        
        self.dataset.set_xtrain(np.array(x_aux))
    
    def CreateModel(self):
        input_shape = self.dataset.get_xtrain().shape
        print("input_shape:     ", input_shape)
        self.model = models.Sequential()
        self.model.add(layers.Conv2D( 64, (3,3), activation='relu', input_shape=(5,7,1) ))
        self.model.add(layers.Conv2D(128, (3,3), activation='relu'))
        self.model.add(layers.Conv2D(256, (3,3), activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))
        self.model.summary()

    def ConvolutionalNeuralNetworks_run(self):
        self.ReshapeDataset()
        self.CreateModel()
    
