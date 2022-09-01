"""
Experiment.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
"""
from tkinter import Image
from Parameters import Parameters
from Dataset import Dataset
from SMOTE import SMOTE
from NeuralNetworks import NeuralNetworks
from DecisionTree import DecisionTree
from SupportVectorMachine import SupportVectorMachine
from ImageConversion import ImageConversion
from ConvolutionalNeuralNetworks import ConvolutionalNeuralNetworks

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
        self.Implementacion(self.getmethod())
        print("Execute experiment!!!")
    
    def getmethod(self):
        return self.method

    def getDataset(self):
        return self.dataset
    
    def Implementacion(self, method):
        if method == "RNN":
            rnn = NeuralNetworks()
            capaOculta1 = [40, 30, 20, 12]
            capaOculta2 = [25, 20, 15, 6]
            capaOculta3 = [12,  8,  4, 3]
            mejor_red_encontrada = rnn.SearchMesh(capaOculta1,capaOculta2,capaOculta3,self.getDataset(),2500)#retorna la mejor red encontrada
            rnn.EvaluateBestNetwork(mejor_red_encontrada[0],mejor_red_encontrada[1],mejor_red_encontrada[2],self.getDataset(),2500)#2 neuronas capa salida
        elif method == "RNN_":
            rnn = NeuralNetworks()
            rnn.SearchConfigNetworkInput(self.getDataset(),2500)
        elif method == "RNN1":
            rnn = NeuralNetworks()
            capaOculta1 = [40, 30, 20, 12]
            capaOculta2 = [25, 20, 15, 6]
            capaOculta3 = [12,  8,  4, 3]
            mejor_red_encontrada = rnn.SearchMesh(capaOculta1,capaOculta2,capaOculta3,self.getDataset(),2500)#retorna la mejor red encontrada
            rnn.EvaluateBestNetwork_One(mejor_red_encontrada[0],mejor_red_encontrada[1],mejor_red_encontrada[2],self.getDataset(),2500)#2 neuronas capa salida
        elif method == "DT":
            dt = DecisionTree(self.getDataset())
            dt.DecisionTreeModel(self.getDataset())
        elif method == "SVM":
            svm = SupportVectorMachine()
            svm.SupportVectorMachineModel(self.getDataset())
        elif method == "CNN":
            #convertionimage = ImageConversion(self.getDataset())
            #convertionimage.Convertion()
            cnn = ConvolutionalNeuralNetworks(self.getDataset())
            cnn.ConvolutionalNeuralNetworks_run()



