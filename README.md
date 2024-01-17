# test_tf_cpu_gpu
Script(s) for testing execution time of Keras-TF ANN training in CPU and GPU
Benchmark_v2.py is the functional script for 

## Dependencies
Before using the script, make sure you have installed numpy, platform, psutil, cpuinfo, and tensorflow

## Usage

benchmark_v2.py [scan, run, show, save]
    Parameters: scan: Scan the hardware's device, return the device name, the processor name(cpu gen), processor speed [ Hz ], device's total RAM, operating system and the GPU if it has one. No argument needed.
                help: -h, displays the help of the program.
                run: Runs the program, it has default values for each of the arguments, the values in the argument can be changed
                     - H: (default 1) The device in which the model is running, 1=CPU, 2=GPU.
                     - d: (default 1) Select the dataset used for training 1=fashion_mnist, 2=cifar10.
                     - l: (default 3) The number of hidden layers used in the model.
                     - i: (default 1000) The epochs of training for the model.
                show: Print all the devices information saved. No argument needed.
                save: Save the current benchmark. No argument needed.