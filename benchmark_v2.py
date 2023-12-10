# Importing zone
# for creating the arguments
import argparse
import sys
# basic data libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# for accessing the hardware info
import platform
import psutil                       #pip install psutil
import cpuinfo                      #pip install py-cpuinfo
# for ml models
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler                                                        # to standarize
from sklearn.decomposition import PCA                                                                   # to standarize
from sklearn.ensemble import RandomForestClassifier                                                     # model
from sklearn.linear_model import LogisticRegression                                                     # model
from sklearn.svm import SVC                                                                             # model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis     # model
# misc libraries
import multiprocessing
import os
from time import time

# Pseudocode

# A code with different options to use in the command line
# scan: (optional) scan the hardware to tell the user if the machine has more than a CPU or a GPU or more
# its a good idea to create a dictionary to save the information of the system
# Training: (default=fminist) Choose the training dataset used for the benchmark
# processor: (default = 1) Choose the hardware in which run the benchmark

# no ponerlo como opcion, hasta el final
# save: (optional, deflault=false) if true save the results in an actualizable list score, need a provided name to identify the score


# Pruebas


# print(multiprocessing.cpu_count())
# print(os.cpu_count())
# print(cpuinfo.get_cpu_info())

# Las metricas que necesito
# print(f"Architecture: {platform.architecture()}")   #Arquitectura ***
# print(f"Network name: {platform.node()}")           #Nombre del equipo ****
# print(f"Operating System: {platform.platform()}")   #Sistema operativo

# cpu_info = cpuinfo.get_cpu_info()
# print(f"CPU name: {cpu_info['brand_raw']}")                  #Nombre del procesador ****
# print(f"CPU speed: {cpu_info['hz_actual_friendly']}")        #Velocidad del procesador
# print(f"CPU data:\n{cpu_info['hz_advertised_friendly']}")
# print(cpu_info.keys())

# print(f"RAM memory: {psutil.virtual_memory().total / 1024 /1024 / 1024: .2f} GB") #RAM ****

# if hasattr(psutil, 'sensors_gpu'):
#     gpu_info = psutil.sensors_gpu()
#     print(f"GPU data:\n{gpu_info}")
# else:
#     print(f"GPU not found")

# Function declaration zone
def scan_hardware():
    # Get CPU information
    cpu_info = cpuinfo.get_cpu_info()
    ram_total = np.round(psutil.virtual_memory().total / 1024 /1024 / 1024, 2)

    attributes = {'system':platform.node(),
                    'cpu_name':cpu_info['brand_raw'],
                    'cpu_speed':cpu_info['hz_actual_friendly'],
                    'cpu_arch':platform.architecture(),
                    'total_ram':ram_total,
                    'gpu':"GPU not found"
                    }

    print(f"System name: {attributes['system']}")
    print(f"CPU data: {attributes['cpu_name']}")
    print(f"CPU speed: {attributes['cpu_speed']}")
    print(f"Architecture: {attributes['cpu_arch']}")
    
    # Get RAM info
    
    print(f"RAM memory: {attributes['total_ram']} GB") #RAM ****
    # podría agregar la información de la ram usada y de la ram disponible

    #  GPU info (if available)
    if hasattr(psutil, 'sensors_gpu'):
        gpu_info = psutil.sensors_gpu()
        attributes['gpu'] = gpu_info
        print(f"GPU data: {gpu_info}")
    else:
        print(attributes['gpu'])
    
    return attributes


# Parser zone

def arguments():
        # Zona de declaración
    parser = argparse.ArgumentParser(
        prog="benchmark_v1.ipynb",
        description="A program for testing the speed training of ML models with tensorflow",
        epilog="This program uses tensorflow and sklearn to train ML models, in the CPU(s) or/and in the GPU(s)")
    
    parser.add_argument("--venv", action="store_true", help="create a virtual environment for easy installing and uninstalling libraries")

    # Crear subparsers
    subparser = parser.add_subparsers(title="subcommands", dest="subcommand", help="program subcommands")

    # Creo los subparsers, para escanear, para mostrar, para correr y para salvar el benchmark
    scan_parser = subparser.add_parser("scan", help="scan the current hardware") # scan no necesita tener niguna opción
    show_subparser = subparser.add_parser("show", help="display the list of saved benchmarks") # maybe argumentos posicionales del primer y último benchmark
    run_subparser = subparser.add_parser("run", help="options for running the current benchmark") # run si tiene que tener argumentos 
    run_subparser.add_argument("-H","--hardware",choices=[1,2], default=1, type=int, help="select the hardware in which the code is ran")
    run_subparser.add_argument("-p","--processor", choices=[1,2,3,4,5,6,7,8], default=1, type=int, help="select the quantity of processors that can be used in the training")
    run_subparser.add_argument("-m","--model",choices=[1,2,3], default=1, type=int, help="select the ML model")
    run_subparser.add_argument("-d","--dataset", choices=[1,2,3,4,5], default=5, type=int, help="select the dataset to train the ML model")
    run_subparser.add_argument("-i","--iteration", action="store", help="Number of repeated trainings for each algorithm (int, default = 1000)",
                        type=int,default=1000,dest='iteration',metavar='Iterations') # change the number to 1000, for practicity i used 100
    save_subparser = subparser.add_parser("save", help="save current benchmark") # quizá un argumento posicional de nombre
    ###### NOTA: revisar el nargs, para que solo pueda admitir un solo argumento
    return parser.parse_args()

def dataset_processing(num):
    scaler = StandardScaler()
    if num==1:
        f_mnist = keras.datasets.fashion_mnist
        # Separate the data from fmnist into train and test data
        (x_train,y_train),(x_test,y_test) = f_mnist.load_data()
        # Copy the data so we can reshape a 2D matrix
        xc_train = x_train.copy()
        yc_train = y_train.copy()
        xc_test = x_test.copy()
        yc_test = y_test.copy()
        nshape = xc_train.shape[1]*xc_train.shape[2]
        xc_train = np.reshape(xc_train,[xc_train.shape[0],nshape])
        xc_test	= np.reshape(xc_test,[xc_test.shape[0],nshape])
        xc_train_std = scaler.fit_transform(xc_train)
        xc_test_std = scaler.transform(xc_test)
        print("The fashion mnist dataset has been chosen")
        return xc_train_std, yc_train, xc_test_std, yc_test
    elif num==2:
        digits = keras.datasets.cifar10
        (x_train,y_train),(x_test,y_test) = digits.load_data()
        # one hot encoding labels
        y_train_encoded = keras.utils.to_categorical(y_train, num_classes = 10, dtype = 'float32')
        y_test_encoded = keras.utils.to_categorical(y_test, num_classes = 10, dtype = 'float32')
        xc_train = x_train.copy()
        yc_train = y_train.copy()
        xc_test = x_test.copy()
        yc_test = y_test.copy()
        xc_train_std = scaler.fit_transform(xc_train)
        xc_test_std = scaler.transform(xc_test)
        print("The digits dataset has been chosen")
        return xc_train_std, yc_train, xc_test_std, yc_test
    elif num==3:
        return (num)
    elif num==4:
        return (num)
    elif num==5:
        return (num)
    
def get_model(value):
    return value

## Necesitamos empezar el programa al menos con un dataset, un modelo de ML y un numero determinado de iteraciones
def main_func():
    args = arguments()
    # print(args.subcommand)
    if args.subcommand == "scan":
        # Aquí vamos a hacer que escanee el hardware de una vez, faltaría mejorarlo para que 
        system = scan_hardware()
    if args.subcommand == "show":
        # Para que muestre el benchmark que llevamos, se deberá guardar en un html o un archivo por separado\
        print("Saved benchmarks")
    if args.subcommand == "run":
        # Para que corra el código con todos los posibles escenarios, deberá correr al menos un modelo con un dataset
        print(args)
        if args.hardware == 1:
            print("Running on CPU")
            # print(args.iteration)
            dataset_processing(args.dataset)
        if args.hardware == 2:
            print("Running on GPU")
    if args.subcommand == "save":
        print("The benchmark has been saved succesfully")

##### NOTA 2: el resultado que queremos observar es unicamemnte en el tiempo de ejecución

if __name__ == "__main__":
    main_func()
  