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

from dtreeplt import dtreeplt
import pandas as pd
from Plotting import Plotting
from IPython import get_ipython
from datetime import datetime


class DecisionTree:
    def __init__(self,dataset):
        self.clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=42,max_depth=4, min_samples_leaf=5)
        self.columns = dataset.get_dataset().columns.values.tolist()
        self.TargetVariable = self.columns.pop()
        self.Predicts = dataset.get_dataset().columns.values.tolist()[:-1]
        self.decisiontree = []
        self.prediction = []
        self.matrizconfusion = []
        self.f = open("logger/DecisionTree_"+str(datetime.now())+".txt",'w')

    def DecisionTreeModel(self,dataset):
        self.decisiontree = self.clf.fit(dataset.get_xtrain(),dataset.get_ytrain())
        self.prediction = self.decisiontree.predict(dataset.get_xtest())
        self.matrizconfusion = metrics.confusion_matrix(dataset.get_ytest(),self.prediction)
        print(metrics.classification_report(dataset.get_ytest(),self.prediction))
        print(self.matrizconfusion)
        print("F1 Score:                  ",metrics.f1_score(dataset.get_ytest(),self.prediction))
        self.f.write("F1 Score:   " + str(metrics.f1_score(dataset.get_ytest(),self.prediction)) + "\n")
        print("Correlación de Mattews:    ",matthews_corrcoef(dataset.get_ytest(),self.prediction))
        self.f.write("Correlación de Mattews:    " + str(matthews_corrcoef(dataset.get_ytest(),self.prediction)) + "\n")

    def SignificativeImportance(self):
        get_ipython().run_line:magin('matplotlib','inline')
        feature_importances = pd.Series(self.decisiontree.feature_importances_, index=self.Predicts)
        self.f.write("Mas significative smportance:\n    " + str(feature_importances) + "\n")
        print(feature_importances)
        feature_importances.nlargest(10).plot(kind='barh')

    def MatrixConfusion(self):
        confusionmatrix = Plotting()
        confusionmatrix.ConfusionMatrix(self.matrizconfusion)

    def CloseFile(self):
        self.f.close()
