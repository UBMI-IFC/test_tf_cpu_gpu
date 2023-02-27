#!/usr/bin/env python3


import argparse

def argument_parser():
    parser = argparse.ArgumentParser(
        prog='test_tf_cpu_gpu.py',
        description='A program for test GPU and CPU installations with tensorflow',
        epilog=':P')

    parser.add_argument('-d', '--device', default='1',
                        help='Device for performing tests. 1) CPU, 2) GPU, 3) both [1]')
    parser.add_argument('-t', '--test', help='Teset dataset'
                        )
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = argument_parser()

    print(args)
    print('-=0----------------------------------------')
    if args.device == '1':
        print('You choose CPU')
    elif args.device == '2':
        print('You choose GPU')
    elif args.device == '3':
        print('You choose both (GPU and CPU)')
