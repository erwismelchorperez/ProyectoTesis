"""
ImageConversion.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre

This class will take care of converting the arrays to binary images
"""
import numpy as np
#from PIL import Image as im
import cv2
class ImageConversion:
    def __init__(self,dataset):
        self.dataset = dataset
    
    def Convertion(self):
        count = 0
        for x in self.dataset.get_xtrain()[:5]:
            x = x*255
            aux = np.reshape(x, (4,6)).astype('uint8')
            print("x["+str(count)+"]:    ", x.shape,"  aux:  ",aux.shape)
            cv2.imwrite("./image/x"+str(count)+".png",aux)
            count = count + 1