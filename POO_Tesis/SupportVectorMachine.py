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
        self.svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
        self.svr_lin = SVR(kernel="linear", C=100, gamma="auto")
        self.svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
    def VerResultados(self,dataset):
        lw = 2
        svrs = [self.svr_rbf, self.svr_lin, self.svr_poly]
        kernel_label = ["RBF", "Linear", "Polynomial"]
        model_color = ["m", "c", "g"]

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
        for ix, svr in enumerate(svrs):
            axes[ix].plot(
                dataset.get_xtrain(),
                svr.fit(dataset.get_xtrain(), dataset.get_ytrain()).predict(dataset.get_xtrain()),
                color=model_color[ix],
                lw=lw,
                label="{} model".format(kernel_label[ix]),
            )
            print(svr.support_,"   ",svr.support_[0])
            axes[ix].scatter(
                dataset.get_xtrain()[svr.support_],
                dataset.get_ytrain()[svr.support_[0]],
                facecolor="none",
                edgecolor=model_color[ix],
                s=50,
                label="{} support vectors".format(kernel_label[ix]),
            )
            axes[ix].scatter(
                dataset.get_xtrain()[np.setdiff1d(np.arange(len(dataset.get_xtrain())), svr.support_)],
                dataset.get_ytrain()[np.setdiff1d(np.arange(len(dataset.get_xtrain())), svr.support_)],
                facecolor="none",
                edgecolor="k",
                s=50,
                label="other training data",
            )
            axes[ix].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.1),
                ncol=1,
                fancybox=True,
                shadow=True,
            )

        fig.text(0.5, 0.04, "data", ha="center", va="center")
        fig.text(0.06, 0.5, "target", ha="center", va="center", rotation="vertical")
        fig.suptitle("Support Vector Regression", fontsize=14)
        plt.show()

