"""
NeuralNetworks.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
"""
from tabnanny import verbose
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix
from Plotting import Plotting
class NeuralNetworks:
    def __init__(self):
        self.hidden_layer_one = 20
        self.hidden_layer_two = 15
        self.hidden_layer_three = 12
        self.metrics = self.get_metrics()
        self.early_stopping = self.get_earlystopping()
        self.model = Sequential()
        self.initializer = self.get_initializer()
        np.random.seed(0)
    
    def set_hiddenlayerone(self,hidden_layer_one):
        self.hidden_layer_one = hidden_layer_one
    def set_hiddenlayertwo(self,hidden_layer_two):
        self.hidden_layer_two = hidden_layer_two
    def set_hiddenlayerthree(self,hidden_layer_three):
        self.hidden_layer_three = hidden_layer_three

    def get_hiddenlayerone(self):
        return self.hidden_layer_one
    def get_hiddenlayertwo(self):
        return self.hidden_layer_two
    def get_hiddenlayerthree(self):
        return self.hidden_layer_three


    def get_metrics(self):
        return [keras.metrics.TruePositives(name='tp'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.TrueNegatives(name='tn'),
                keras.metrics.FalseNegatives(name='fn'), 
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='prc', curve='PR')]

    def get_earlystopping(self):
        return tf.keras.callbacks.EarlyStopping(monitor='val_prc', verbose=1,patience=10,mode='max',restore_best_weights=True)
    
    def get_initializer(self):
        self.initializer = tf.keras.initializers.GlorotNormal()

    def CreateModel(self,numclass,feature_vector_length):
        self.model = Sequential()
        self.model.add(Dense(self.get_hiddenlayerone(), kernel_initializer=self.get_initializer(), input_shape=(feature_vector_length,), activation='tanh'))
        self.model.add(Dense(self.get_hiddenlayertwo(), activation='tanh'))
        self.model.add(Dense(self.get_hiddenlayerthree(), activation='tanh'))
        self.model.add(Dense(numclass, activation='sigmoid'))

    def CreateModelOne(self,numclass,feature_vector_length):
        self.model = Sequential()
        self.model.add(Dense(self.get_hiddenlayerone(), kernel_initializer=self.get_initializer(), input_shape=(feature_vector_length,), activation='sidmoid'))
        self.model.add(Dense(self.get_hiddenlayertwo(), activation='sidmoid'))
        self.model.add(Dense(self.get_hiddenlayerthree(), activation='sidmoid'))
        self.model.add(Dense(numclass, activation='sigmoid'))

    def CompileModel(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam',metrics=self.get_metrics())
        
    
    def TrainModel(self,x_train,y_train,_epochs,x_validation,y_validation):
        self.model.fit(x_train,y_train,epochs=_epochs,batch_size=250,verbose=1,validation_split=0.2,callbacks=[self.get_earlystopping()],validation_data=(x_validation, y_validation))

    def EvaluateModel(self,x_test,y_test):
        return self.model.evaluate(x_test,y_test,verbose = 0)
    
    def PredictModel(self,x_test):
        return self.model.predict(x_test)

    #función que realizará la búsqueda en malla
    def SearchMesh(self,hiddenlayer_one,hiddenlayer_two,hiddenlayer_three,dataset,epochs):#realizaremos la búsqueda en malla
        hiddenlayer_one = np.array(hiddenlayer_one)
        hiddenlayer_two = np.array(hiddenlayer_two)
        hiddenlayer_three = np.array(hiddenlayer_three)

        """Resultados búsqueda en malla"""
        mejor_red_encontrada = []
        mejor_matrizconfusion = []
        mejor_accuracy = 0

        for capa1 in hiddenlayer_one:
            for capa2 in hiddenlayer_two:
                for capa3 in hiddenlayer_three:
                    self.set_hiddenlayerone(capa1)
                    self.set_hiddenlayertwo(capa2)
                    self.set_hiddenlayerthree(capa3)

                    self.CreateModel(dataset.getNumClass(),dataset.getfeature_vector_length())#creación del modelo
                    self.CompileModel()#compilación del modelo
                    #print(dataset.getNumClass(),"      ",dataset.getfeature_vector_length(),"     ",dataset.get_xtrain().shape,"    ",dataset.get_ytrain().shape,"     ",epochs,"    ",dataset.get_xvalidation().shape,"     ",dataset.get_yvalidation().shape)
                    self.TrainModel(dataset.get_xtrain(),dataset.get_ytrainCategorical(),epochs,dataset.get_xvalidation(),dataset.get_yvalidationCategorical())#entrenamiento del modelo

                    test_result = self.EvaluateModel(dataset.get_xtest(),dataset.get_ytestCategorical())

                    if mejor_accuracy < test_result[1]:
                        mejor_accuracy = test_result[1]
                        mejor_red_encontrada = [capa1,capa2,capa3]
                        mejor_matrizconfusion = confusion_matrix(dataset.get_ytestCategorical().argmax(axis=1), self.PredictModel(dataset.get_xtest()).argmax(axis=1))

        print("Termina búsqueda en malla!!!")        
        print(mejor_accuracy)
        print(mejor_red_encontrada)
        print(mejor_matrizconfusion)
        return mejor_red_encontrada

    def EvaluateBestNetwork(self,hiddenlayer_one,hiddenlayer_two,hiddenlayer_three,dataset,epochs):
        self.set_hiddenlayerone(hiddenlayer_one)
        self.set_hiddenlayertwo(hiddenlayer_two)
        self.set_hiddenlayerthree(hiddenlayer_three)

        self.CreateModel(dataset.getNumClass(),dataset.getfeature_vector_length())#creación del modelo
        self.CompileModel()#compilación del modelo
        self.TrainModel(dataset.get_xtrain(),dataset.get_ytrainCategorical(),epochs,dataset.get_xvalidation(),dataset.get_yvalidationCategorical())#entrenamiento del modelo

        test_result = self.EvaluateModel(dataset.get_xtest(),dataset.get_ytestCategorical())
        matrizconfusion = confusion_matrix(dataset.get_ytestCategorical().argmax(axis=1), self.PredictModel(dataset.get_xtest()).argmax(axis=1))
        confusionmatrix = Plotting()
        confusionmatrix.ConfusionMatrix(matrizconfusion)

    def EvaluateBestNetwork_One(self,hiddenlayer_one,hiddenlayer_two,hiddenlayer_three,dataset,epochs):
        self.set_hiddenlayerone(hiddenlayer_one)
        self.set_hiddenlayertwo(hiddenlayer_two)
        self.set_hiddenlayerthree(hiddenlayer_three)

        self.CreateModelOne(dataset.getNumClass()-1,dataset.getfeature_vector_length())#creación del modelo
        self.CompileModel()#compilación del modelo
        self.TrainModel(dataset.get_xtrain(),dataset.get_ytrain(),epochs,dataset.get_xvalidation(),dataset.get_yvalidation())#entrenamiento del modelo

        test_result = self.EvaluateModel(dataset.get_xtest(),dataset.get_ytest())
        respuestas = self.PredictModel(dataset.get_xtest()).argmax(axis=1)
        res = respuestas.reshape((respuestas.shape[0]))
        res = (res>0.5)*1
        matrizconfusion = confusion_matrix(dataset.get_ytest(), res)
        confusionmatrix = Plotting()
        confusionmatrix.ConfusionMatrix(matrizconfusion)

    def SearchConfigNetworkInput(self,dataset,epochs):
        r = np.power((dataset.get_xtrain().shape[1]/2),0.25)
        self.set_hiddenlayerone(int(2*np.power(r,3)))
        r = np.power(self.get_hiddenlayerone()/2,0.25)
        self.set_hiddenlayertwo(int(2*np.power(r,2)) + 2)
        r = np.power((self.get_hiddenlayertwo()/2),0.25)
        self.set_hiddenlayerthree(int(2*r)+1)

        self.CreateModel(dataset.getNumClass(),dataset.getfeature_vector_length())#creación del modelo
        self.CompileModel()#compilación del modelo
        self.TrainModel(dataset.get_xtrain(),dataset.get_ytrainCategorical(),epochs,dataset.get_xvalidation(),dataset.get_yvalidationCategorical())#entrenamiento del modelo

        test_result = self.EvaluateModel(dataset.get_xtest(),dataset.get_ytestCategorical())
        matrizconfusion = confusion_matrix(dataset.get_ytestCategorical().argmax(axis=1), self.PredictModel(dataset.get_xtest()).argmax(axis=1))
        confusionmatrix = Plotting()
        confusionmatrix.ConfusionMatrix(matrizconfusion)








