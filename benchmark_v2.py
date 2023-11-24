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
    show_subparser = subparser.add_parser("show", help="display the list of saved benchmarks") # show tampoco necesita tener ninguna opción
    run_subparser = subparser.add_parser("run", help="options for running the current benchmark") # run si tiene que tener argumentos 
    run_subparser.add_argument("-i","--iter", help="Number of repeated trainings for each algorithm (int, default = 1000)",
                        type=int,default=100,dest='iter',metavar='Iterations') # change the number to 1000, for practicity i used 100
    run_subparser.add_argument("-d","--dataset", choices=[1,2,3,4,5], default=1, help="select the dataset to train the ML model")
    run_subparser.add_argument("-m","--model",choices=[1,2,3], default=1, help="select the ML model")
    run_subparser.add_argument("-p","--processor", choices=[1,2,3,4,5,6,7,8], default=1, help="select the quantity of processors that can be used in the training")
    run_subparser.add_argument("-h","--hardware",choices=[1,2], default=1, help="select the hardware in which the code is ran")
    save_subparser = subparser.add_parser("save", help="save current benchmark") # save tampoco necesita argumentos
    
    return parser.parse_args()

# if __name__ == "__main__":
    ## Este es el que será el código final, pero estamos entendiendo el argparser
def main_func():
    args = arguments()
    # print(args.subcommand)
    if args.subcommand == "scan":
        system = scan_hardware()
    if args.subcommand == "show":
        print("benchmark salvados")
    if args.subcommand == "run":
        print("corriendo el programa")
    if args.subcommand == "save":
        print("El archivo se ha guardado correctamente")




main_func()
