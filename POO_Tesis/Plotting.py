"""
Plotting.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
"""
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_confusion_matrix

class Plotting:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.line = None
        self.annot = None
        self.z_labels = None
    
    def ConfusionMatrix(self,confusionmatrix):
        self.fig, self.ax = plot_confusion_matrix(conf_mat=confusionmatrix)
        plt.show()