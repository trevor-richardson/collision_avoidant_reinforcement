import time
import numpy as np
import random
import math
import sys
import scipy.io as sio
import os
from os.path import isfile, join
from os import listdir

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')

base_dir = config['DEFAULT']['BASE_DIR']
sys.path.append(base_dir + '/vrep_scripts/')

from test_baselines import execute_exp

parser = argparse.ArgumentParser(description='Test Baselines')
parser.add_argument('--action_arg', type=int, default=-1, metavar='N',
                    help='Action argument')
parser.add_argument('--validation_iterations', type=int, default=50, metavar='N',
                    help='Number of times I want to validate a reinforcement learning model')
parser.add_argument('--num_seeds', type=int, default=10, metavar='N',
                    help='Number of sets I want to collect distribution over models')
args = parser.parse_args()


def main():
    results_lst = []

    for index in range(args.num_seeds):
        print(index, "####################################################################################################################\n")

        count = 0
        for inner_index in range(args.validation_iterations):
            state, collision_detector = execute_exp(args.action_arg)

            if collision_detector > 0:
                print("Hit")
                count+=1
            else:
                print("Miss")


        print("count ", count)
        results_lst.append([index, count])
    np.save("valbaseline_results_action" + str(args.action_arg), np.asarray(results_lst))


if __name__ == '__main__':
    main()
