"""
main.py
Autor: Erwis Melchor Pérez
Maestría en Tecnologías de Cómputo Aplicado
Tesis
Tercer Semestre
"""
import sys
from Experiment import Experiment

if len(sys.argv) < 2:
    print("Syntax error!")
    exit(0)

method = sys.argv[1]
params_file = sys.argv[2]
print("method: ",method,"    params_file:   ",params_file)
experiment = Experiment(method,params_file)
experiment.execute_experiment()