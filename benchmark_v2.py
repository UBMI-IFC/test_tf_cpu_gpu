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
import csv
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

def dataset_processing(num):
    if num==1:
        print("The fashion mnist dataset has been chosen")
        dataset = keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = dataset.load_data()

        # checking images
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        
        # scaling, from 8bit pixels to 0-1 values
        x_train_std = x_train / 255
        x_test_std = x_test / 255
        dataset_name = "fashion_mnist"
        return x_train_std, y_train, x_test_std, y_test, dataset_name
    
    elif num==2:
        print("The digits dataset has been chosen")
        dataset = keras.datasets.cifar10
        
        # Separate train and test dataset
        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        
        # Scaling the image, transform data from 8bit values to 0-1 range
        x_train_std = x_train/255
        x_test_std = x_test/255
        
        # one hot encoding labels
        y_train_encoded = keras.utils.to_categorical(y_train, num_classes = 10, dtype = "float32")
        y_test_encoded = keras.utils.to_categorical(y_test, num_classes = 10, dtype = "float32")
        dataset_name = "cifar10"
        return x_train_std, y_train_encoded, x_test_std, y_test_encoded, dataset_name
    
    elif num==3:
        # This option is not implementated yet
        return (num)
    
    elif num==4:
        # This option is not implementated yet
        return (num)
    
    elif num==5:
        # This option is not implementated yet
        return (num)
    
def get_model(in_shape, dense_neurons, hidden_layers=1):
    """This function is for convenience. We can create exactly the same model easly changing the device we are going to use.
    in_shape      is the input shape of the dataset used
    dense_neurons is the quantity of neurons per dense layer
    hidden_layers is the number of dense layers the model has"""
    # Flatten layer for input
    # Create list of layers, first one is a flatten layer
    layers = [keras.layers.Flatten(input_shape=in_shape)]
    # hideen layers, how many?
    lesser_layers = np.round(dense_neurons/hidden_layers)
    for i in range(hidden_layers):
        if i == 0:
            layers.append(keras.layers.Dense(dense_neurons, activation="relu"),)
        else:
            dense_neurons -= lesser_layers
            layers.append(keras.layers.Dense(dense_neurons, activation="relu"),)
    # output layer, classification
    layers.append(keras.layers.Dense(10, activation="sigmoid"))
    # creating model 
    model = keras.Sequential(layers)
    # compiling and selection optimizer and loss funcition
    model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    return model

def save_benchmark(lst):
    with open("benchmark.csv", "a",newline="",encoding="utf-8") as benchmark:
        writer_obj = csv.writer(benchmark)
        writer_obj.writerow(lst)
        benchmark.close()
    return None

def show_benchmark():
    with open("benchmark.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            print(row)

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
    run_subparser.add_argument("-d","--dataset", choices=[1,2,3,4,5], default=1, type=int, help="select the dataset to train the ML model")
    run_subparser.add_argument("-l","--layers", default=3, type=int, help="select the number of hidden layers to train the model")
    run_subparser.add_argument("-i","--iteration", action="store", help="Number of repeated trainings for each algorithm (int, default = 1000)",
                        type=int,default=1000,dest='iteration',metavar='Iterations') # change the number to 1000, for practicity i used 100
    save_subparser = subparser.add_parser("save", help="save current benchmark") # quizá un argumento posicional con el nombre con el que se va a guardar
    ###### NOTA: revisar el nargs, para que solo pueda admitir un solo argumento
    return parser.parse_args()

## Necesitamos empezar el programa al menos con un dataset, un modelo de ML y un numero determinado de iteraciones
def main_func():
    args = arguments()
    # print(args.subcommand)
    if args.subcommand == "scan":
        # Aquí vamos a hacer que escanee el hardware de una vez, faltaría mejorarlo para que lea otras gpus que no sean de nvidia 
        # y que diga si hay más de una gpu y más de un cpu
        system = scan_hardware()
    if args.subcommand == "show":
        # Para que muestre el benchmark que llevamos, se deberá guardar en un html o un .csv por separado
        print("Saved benchmarks")
        show_benchmark()
    if args.subcommand == "run":
        # Para que corra el código con todos los posibles escenarios, deberá correr al menos un modelo con un dataset
        print(args)
        x_train, y_train, x_test, y_test, data_name = dataset_processing(args.dataset)
        dshape = x_train.shape[1:]
        dense = np.prod(dshape)
        if args.hardware == 1:
            print("Running on CPU")
            t1 = time()
            with tf.device("/CPU:0"):
                # use 5 layers
                cpu_model = get_model(dshape, dense, hidden_layers=args.layers)
                cpu_model.fit(x_train, y_train, epochs=args.iteration)
            t2 = time()
            test_time = t2-t1
            cpu_gpu = "CPU"
            print(f"Test CPU:  {test_time} seconds")
        if args.hardware == 2:
            print("Running on GPU")
            try:
                t1 = time()
                with tf.device("/GPU:0"):
                    # use 5 layers
                    gpu_model = get_model(dshape, dense, hidden_layers=args.layers)
                    gpu_model.fit(x_train, y_train, epochs=args.iteration)
                t2 = time()
                test_time = t2-t1
                cpu_gpu = "GPU"
                print(f"Test GPU:  {test_time} seconds")
            except:
                print(system["gpu"])
    if args.subcommand == "save":
        # write in the csv file for saving
        # labels = ["System_name","CPU_data","RAM_total","GPU_data","CPU/GPU","Dataset","#_layers","Epochs","Training_time[s]"]
        attributes = [system["system"],system["cpu_name"],system["total_ram"],system["gpu"],cpu_gpu,data_name,args.layers,args.iteration,test_time]
        save_benchmark(attributes)
        print("The benchmark has saved succesfully")

if __name__ == "__main__":
    main_func()
