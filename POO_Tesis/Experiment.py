"""
Experiment.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
"""
#from tkinter import Image
from XGBoost import XGBoost_,XGBoost_Hyperparameters
from Parameters import Parameters
from Dataset import Dataset
from SMOTE import SMOTE
from NeuralNetworks import NeuralNetworks
from DecisionTree import DecisionTree
from SupportVectorMachine import SupportVectorMachine, SupportVectorMachineV2
from ImageConversion import ImageConversion
from ConvolutionalNeuralNetworks import ConvolutionalNeuralNetworks, ConvolutionalNeuralNetworksMnist

class Experiment:
    def __init__(self, method, params_file):
        self.method = method
        self.parameters = Parameters(params_file)
        self.dataset = Dataset(self.parameters,self.method)
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
            rnn.WriteLogger("RNN Busqueda en Malla 2 Neuronas")
            capaOculta1 = [40, 30, 20]
            capaOculta2 = [25, 20, 15]
            capaOculta3 = [12,  8,  4]
            funcionactivacion1 = ['tanh','sigmoid','relu','softmax','selu']
            funcionactivacion2 = ['tanh','sigmoid','relu','softmax','selu']
            funcionactivacion3 = ['tanh','sigmoid','relu','softmax','selu']
            funcionactivacion4 = ['tanh','sigmoid','relu','softmax','selu']
            """
            capaOculta1 = [40, 30, 20, 12, 15, 25, 35, 50, 45, 18, 22, 24, 32, 36, 38, 44, 66, 88]
            capaOculta2 = [25, 20, 15,  6,  8, 10, 14, 16, 18, 22, 24, 30, 44, 40, 66, 56, 52, 28]
            capaOculta3 = [12,  8,  4,  3,  2,  6, 10, 14, 20, 22, 26, 32, 24, 46, 40, 62, 48, 88]
            """

            mejor_red_encontrada = rnn.SearchMesh(capaOculta1,capaOculta2,capaOculta3,self.getDataset(),2500,funcionactivacion1,funcionactivacion2,funcionactivacion3,funcionactivacion4)#retorna la mejor red encontrada
            rnn.EvaluateBestNetwork(mejor_red_encontrada[0],mejor_red_encontrada[1],mejor_red_encontrada[2],self.getDataset(),2500)#2 neuronas capa salida

        elif method == "RNN_":
            rnn = NeuralNetworks()
            rnn.SearchConfigNetworkInput(self.getDataset(),2500)
        elif method == "RNN1":
            rnn = NeuralNetworks()
            rnn.WriteLogger("RNN Busqueda en Malla 1 Neurona")
            capaOculta1 = [40, 30, 20]
            capaOculta2 = [25, 20, 15]
            """
            capaOculta3 = [12,  8,  4]
            capaOculta1 = [40, 30, 20, 12, 15, 25, 35, 50, 45, 18, 22, 24, 32, 36, 38, 44, 66, 88]
            capaOculta2 = [25, 20, 15,  6,  8, 10, 14, 16, 18, 22, 24, 30, 44, 40, 66, 56, 52, 28]
            capaOculta3 = [12,  8,  4,  3,  2,  6, 10, 14, 20, 22, 26, 32, 24, 46, 40, 62, 48, 88]
            """
            mejor_red_encontrada = rnn.SearchMeshOne(capaOculta1,capaOculta2,capaOculta3,self.getDataset(),2500)#retorna la mejor red encontrada
            rnn.EvaluateBestNetwork_One(mejor_red_encontrada[0],mejor_red_encontrada[1],mejor_red_encontrada[2],self.getDataset(),2500)#2 neuronas capa salida
        elif method == "DT":
            dt = DecisionTree(self.getDataset())
            dt.DecisionTreeModel(self.getDataset())
            dt.SignificativeImportance()
            dt.MatrixConfusion()
            dt.CloseFile()
        elif method == "SVM":
            svm = SupportVectorMachine()
            svm.SupportVectorMachineModel(self.getDataset())
            svm = SupportVectorMachineV2()
            svm.TrainModel(self.getDataset())
        elif method == "CNN":
            #convertionimage = ImageConversion(self.getDataset())
            #convertionimage.Convertion()
            cnn = ConvolutionalNeuralNetworks(self.getDataset())
            cnn.ConvolutionalNeuralNetworks_run()
        elif method == "CNNMnist":
            cnn = ConvolutionalNeuralNetworksMnist()
            cnn.ConvolutionalNeuralNetworksMnist_run()
        elif method == 'XGBR':
            xgbr = XGBoost_(self.dataset)
            xgbr.Regression()
            xgbr.Classification()
            xgbr.ClassifierXGBClassifier()

            xgbr.CloseArchivo()
        elif method == 'XGBR_H':
            xgbh = XGBoost_Hyperparameters()
            xgbh.get_dataset()
