"""
SMOTE.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
Aplicación de sobremuestreo de la base de datos en relación a la base minoritaria
"""
from scipy.spatial import distance
import random
import numpy as np
class SMOTE:
    def __init__(self,dataset):
        self.x_train, self.y_train = self.FuncionSMOTE(dataset.get_xtrain(),dataset.get_ytrain())
    
    def FuncionSMOTE(self,x_train,y_train):
        buenos = format(sum(y_train==0))
        malos  = format(sum(y_train==1))
        nuevos  = format(sum(y_train==2))
        aleatorio = random.randint(0, len(x_train)-1)
        #print("dimension:   ", len(x_train), "    ",aleatorio,"  ",y_train.shape)
        
        xnew = []#nuevo elemento
        xi = []#elementos seleccionado
        xik = []# vecino más cercano
        flag = True
        y_trainnew = y_train
        y_trainnew = y_trainnew.to_numpy()
        while flag:
            #generarmos un número aleatorio para verificar que sea de la clase minoritaria
            aleatorio = random.randint(0, len(y_train)-1)
            min_class = y_train.tolist()[aleatorio]
            if min_class == 1:
                new = self.VecinoCercano(aleatorio,x_train,y_train)
                xnew = x_train[aleatorio] + (x_train[new[1]] - x_train[aleatorio])*random.random()
                x_train = np.append(x_train, [xnew])
                x_trainnew = [x_train[i:i+xnew.shape[0]] for i in range(0,len(x_train),xnew.shape[0])]
                x_trainnew = np.array(x_trainnew)
                
                y_trainnew = np.append(y_trainnew,1)
                x_train = x_trainnew
                y_train = y_trainnew
                
            buenos = format(sum(y_train==0))
            malos  = format(sum(y_train==1))
            nuevos  = format(sum(y_train==2))
            
            suma = int(malos) + int(nuevos)
            
            if int(buenos) == suma:
                flag = False
        
        return x_train,y_train
    
    def VecinoCercano(self,aleatorio,x_train,y_train):#vamos a buscar al vecino más cercano
        menordistancia = 100000
        position_menordistancia = 0
        cont = 0
        cont1 = 0
        xik = []
        positionvecinocercano = 0
        xi = x_train[aleatorio]
        arr_menordistancia = []
        for x in x_train:
            salida = y_train.tolist()[cont]
            if salida == 1 and cont != aleatorio:
                #hallar menor dimension:
                d = distance.euclidean(xi, x)#calcular la distancia euclideana para encontrar el menor y este será el vecino más cercano
                if d < menordistancia:
                    menordistancia = d
                    position_menordistancia = cont
                    arr_menordistancia.append([menordistancia,cont])
            
            cont = cont + 1
        return [menordistancia,position_menordistancia]

    def get_xtrain(self):
        return self.x_train
    
    def get_ytrain(self):
        return self.y_train