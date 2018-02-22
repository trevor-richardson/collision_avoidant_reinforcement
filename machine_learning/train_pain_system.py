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
sys.path.append(base_dir + '/machine_learning/data_loaders/')
sys.path.append(base_dir + '/vrep_scripts/')

from deep_dynamics import Deep_Dynamics
from policy_network import Policy_Network
from collision_avoidance import Custom_Spatial_Temporal_Anticipation_NN
from ca_data_loader import VideoDataGenerator
from dd_data_loader import DeepDynamicsDataLoader
from policy_net_data_loader import PolicyNetDataLoader
from run_vrep_simulation import execute_exp


# first goal is to design and load all models ********************
# first goal is to design and load all data loaders *********************

# load make sure to be able to execute the simulation call for generating data
# test i can move data there and make sure the loader works

# make global variables and that models can be run with a script
# make sure all models can be loaded with gpu support

# second goal is to be able to get the data collection running from this script

# third goal is to get the movement of data to new data files from this file

# fourth goal is to design the flow of the algorithm itself on how it updates and loads

# Get full algorithm running with simple -1, 1 reward schema

def main():
    execute_exp(0,10)


if __name__ == '__main__':
    main()
