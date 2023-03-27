# Importing zone
import argparse

# Pseudocode

# A code with different options to use in the command line
# scan: (optional) scan the hardware to tell the user if the machine has more than a CPU or a GPU or more
# Training: (default=fminist) Choose the training dataset used for the benchmark
# processor: (default = 1) Choose the hardware in which run the benchmark
# save: (optional, deflault=false) if true save the results in an actualizable list score, need a provided name to identify the score


# Function declaration zone

def argument_parser():
        # Zona de declaración
    parser = argparse.ArgumentParser(
        prog='Benchmark_version_01.py',
        description='A program for testing the speed training of ML models with tensorflow',
        epilog='Tell hoy to use the program to the user')
    parser.add_argument("")
    return ('...')



if __name__ == "__main__":
    ## Este es el que será el código final, pero estamos entendiendo el argparser
    print("xxxxxxx")

