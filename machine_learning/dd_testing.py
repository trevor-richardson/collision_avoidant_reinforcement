'''Train Reinforcement Learning Policy Network that Learns to Avoid External Pertubation "Pain"'''
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
sys.path.append(base_dir + '/machine_learning/deep_learning_models/')
sys.path.append(base_dir + '/machine_learning/semisup_collision_calc/')
sys.path.append(base_dir + '/vrep_scripts/')

from deep_dynamics import Deep_Dynamics
from test_dd_sim import execute_exp
from pertubation_detection import *

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

'''
----------------------
Read and implement ddpg
'''

''' Global Variables of Interest '''
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Evaluate trained deep dynamics model against the ground truth for collision detection on actual scenario')
parser.add_argument('--policy_inp_type', type=int, default=0, metavar='N',
                    help='Type of input for policy net')

#training and testing args
parser.add_argument('--validation_iterations', type=int, default=50, metavar='N',
                    help='Number of times I want to validate a reinforcement learning model')
parser.add_argument('--num_forward_passes', type=int, default=64, metavar='N',
                    help='Number of forward passes for dropout at test time for multivariate_normal pdf calc')
args = parser.parse_args()

dd_inp_shape = (13)
dd_output_shape = (12)

dd_model = Deep_Dynamics(dd_inp_shape, 60, 40, 30, 20, 20, dd_output_shape)
if torch.cuda.is_available():
    print("Using GPU acceleration")
    dd_model.cuda()

dd_optimizer = torch.optim.Adam(dd_model.parameters(), lr=.001)

def load_dd_model(file_path):
    global dd_model
    print(file_path)
    try:
        dd_model.load_state_dict(torch.load(file_path))
    except ValueError:
        print("Not a valid model to load")
        sys.exit()


mypath = base_dir + '/machine_learning/saved_models/test_dd/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
paths = []

for element in onlyfiles:
    paths.append(base_dir + '/machine_learning/saved_models/test_dd/' + element)

def main():
    global dd_model
    global dd_optimizer
    results_lst = []

    for path in paths:
        print("####################################################################################################################\n")
        results = []
        load_dd_model(path)
        for inner_index in range(args.validation_iterations):

            state, collision_detector = execute_exp()
            rew = dd_test(dd_model, args.num_forward_passes, state[0])
            dd_optimizer.zero_grad()
            if collision_detector > 0:
                print("Hit")
                results_lst.append([1, *rew])
            else:
                print("Miss")
                results_lst.append([0, *rew])

            print(results_lst[-1])
            print("Max DD value ", max(rew))

        f = path.split('/')[-1]
        np.save(f, np.asarray(results_lst))


if __name__ == '__main__':
    main()
