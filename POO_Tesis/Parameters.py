"""
Parameters.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
Parametros de configuración para realizar las pruebas de los modelos
"""
class Parameters:
    def __init__(self,params_file):
        file = open(params_file,"rt")
        for line in file.readlines():
            fragments = line.split("=")
            if fragments[0] == "dataset":
                self.dataset = fragments[1].strip()
            elif fragments[0] == "train":
                self.train = float(fragments[1].strip())
            elif fragments[0] == "test":
                self.test = float(fragments[1].strip())
            elif fragments[0] == "validation":
                self.validation = float(fragments[1].strip())
            elif fragments[0] == "separator":
                self.separator = fragments[1].strip()
                
    def get_dataset(self):
        return self.dataset
    
    def get_train(self):
        return self.train
    
    def get_test(self):
        return self.test
    
    def get_validation(self):
        return self.validation
    
    def get_separator(self):
        return self.separator
        