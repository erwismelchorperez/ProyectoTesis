"""
DecisionTree.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
"""
from tabnanny import verbose
import tensorflow
import tensorflow as tf
from tensorflow import keras
from sklearn import tree

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
class DecisionTree:
    def __init__(self,dataset):
        self.clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=42,max_depth=4, min_samples_leaf=5)
        self.columns = dataset.get_dataset().columns.values.tolist() 
        self.TargetVariable = self.columns.pop()
        self.Predicts = dataset.get_dataset().columns.values.tolist()[:-1]
        self.decisiontree = []
        self.prediction = []
    
    def DecisionTreeModel(self,dataset):
        self.decisiontree = self.clf.fit(dataset.get_xtrain(),dataset.get_ytrain())
        self.prediction = self.decisiontree.predict(dataset.get_xtest())
        print(metrics.classification_report(dataset.get_ytest(),self.prediction))
        print(metrics.confusion_matrix(dataset.get_ytest(),self.prediction))
        print("F1 Score:                  ",metrics.f1_score(dataset.get_ytest(),self.prediction))
        print("Correlación de Mattews:    ",matthews_corrcoef(dataset.get_ytest(),self.prediction))

