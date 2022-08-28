"""
Experiment.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
"""
from Parameters import Parameters
from Dataset import Dataset
from SMOTE import SMOTE
class Experiment:
    def __init__(self, method, params_file):
        self.method = method
        self.parameters = Parameters(params_file)
        self.dataset = Dataset(self.parameters)
        self.smote = SMOTE(self.dataset)

    def execute_experiment(self):
        print("Número de clases:     ",self.dataset.getSalidas())
        self.dataset.set_xtrain(self.smote.get_xtrain())
        self.dataset.set_ytrain(self.smote.get_ytrain())
        self.dataset.ImprimirShape()

        print("Execute experiment!!!")