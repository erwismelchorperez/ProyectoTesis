"""
SupportVectorMachine.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
"""
from tabnanny import verbose
import tensorflow
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import svm
from sklearn.svm import SVR
from Plotting import Plotting
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
class SupportVectorMachine:
    def __init__(self):
        #self.clf = svm.SVC(C=2, kernel='rbf', gamma='scale',cache_size=1000, class_weight='balanced' )
        self.clf = svm.SVC(C=0.001, kernel='rbf', degree= 2, gamma='scale',cache_size=1000, class_weight='balanced',
                        coef0=0.0, decision_function_shape='ovr',max_iter=-1,probability=False, random_state=None,shrinking=True, tol=0.001, verbose=False )
        self.supportvectormachine = []
        self.prediction = []
        self.confusion_matrix = []
    
    def SupportVectorMachineModel(self,dataset):
        self.supportvectormachine = self.clf.fit(dataset.get_xtrain(),dataset.get_ytrain())
        self.prediction = self.supportvectormachine.predict(dataset.get_xtest())
        print(metrics.classification_report(dataset.get_ytest(),self.prediction))
        self.confusion_matrix = metrics.confusion_matrix(dataset.get_ytest(),self.prediction)
        print("F1 Score:    ",metrics.f1_score(dataset.get_ytest(), self.prediction, average='weighted'))
        confusionmatrix = Plotting()
        confusionmatrix.ConfusionMatrix(self.confusion_matrix)

class SupportVectorMachineV2:
    def __init__(self):
        self.parameters = {'C':[0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
        self.svc = svm.SVC()
        self.clf = GridSearchCV(self.svc, self.parameters)
    
    def TrainModel(self,dataset):
        self.clf.fit(dataset.get_xtrain(),dataset.get_ytrain())
        grid = GridSearchCV(estimator=SVR(), param_grid=self.parameters, refit=True, verbose=2)
        grid.fit(dataset.get_xtrain(),dataset.get_ytrain())
        print("       GridSearchCV      ",sorted(self.clf.cv_results_.keys()))
        print("      ", grid.best_params_, "      ", grid.best_estimator_,"    ",grid.best_score_)


