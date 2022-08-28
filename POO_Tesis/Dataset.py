"""
Dataset.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
Importación de la base de datos
"""
from os import sep
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler     #min max
class Dataset:
    def __init__(self,parameters):
        self.dataset = pd.read_csv("./dataset/"+parameters.get_dataset())#lectura del archivo que contiene el csv
        self.dataset['salida'] = self.dataset['salida']-1
        self.numclass = len(pd.unique(self.dataset['salida']))
        self.salidas = self.dataset['salida'] 
        self.entradas = self.Normalizacion_MinMax(self.dataset.drop(["salida"],axis=1))
        (self.x_train,self.x_test,self.x_validation,self.y_train,self.y_test,self.y_validation) = self.Separation_Dataset(parameters)

    def Normalizacion_MinMax(self,entradas):
        sc = MinMaxScaler()
        sc.fit(entradas)
        self.entradas = sc.transform(entradas)
        return self.entradas
    
    def Separation_Dataset(self,parameters):
        x_train, x_test, y_train, y_test = train_test_split(self.entradas, self.salidas, test_size=parameters.get_test(), stratify=self.salidas)
        x_train, x_validation, y_train,y_validation = train_test_split(x_train, y_train, test_size=parameters.get_validation(), stratify=y_train)
        return (x_train,x_test,x_validation,y_train,y_test,y_validation)

    
    def get_dataset(self):
        return self.dataset
    
    def get_xtrain(self):
        return self.x_train
    
    def set_xtrain(self,x_train):
        self.x_train = x_train
    
    def get_ytrain(self):
        return self.y_train
    
    def set_ytrain(self,y_train):
        self.y_train = y_train

    def getSalidas(self):
        return self.numclass

    def ImprimirShape(self):
        print("x_train: ",self.x_train.shape," y_train: ",self.y_train.shape)
        print("x_validation: ",self.x_validation.shape," y_validation: ",self.y_validation.shape)
        print("x_test: ",self.x_test.shape," y_test: ",self.y_test.shape)