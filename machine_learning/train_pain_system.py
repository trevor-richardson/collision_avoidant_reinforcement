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
sys.path.append(base_dir + '/machine_learning/training_classes/')
sys.path.append(base_dir + '/vrep_scripts/')

from deep_dynamics import Deep_Dynamics
from policy_network import Policy_Network
from collision_avoidance import Custom_Spatial_Temporal_Anticipation_NN
from ca_data_loader import VideoDataGenerator
from dd_data_loader import DeepDynamicsDataLoader
from policy_net_data_loader import PolicyNetDataLoader
from run_vrep_simulation import execute_exp
from train_dd import *
from train_anticipation import *

''' Global Variables of Interest '''
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Reinforcement Learning Guided by Deep Dynamics and Anticipation Models in Pytorch')

#training and testing args
parser.add_argument('--training_iterations', type=int, default=20, metavar='N',
                    help='Number of times I want to train a reinforcement learning model')
parser.add_argument('--num_forward_passes', type=int, default=64, metavar='N',
                    help='Number of forward passes for dropout at test time for multivariate_normal pdf calc')
parser.add_argument('--exp_iteration', type=int, default=64, metavar='N',
                    help='Batch size default size is 64')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='Number of hit and number of miss videos (default 64)')
parser.add_argument('--no_filters_0', type=int, default=40, metavar='N',
                    help='Number of activation maps to in layer 0 (default 40)')
parser.add_argument('--no_filters_1', type=int, default=30, metavar='N',
                    help='Number of activation maps to in layer 1 (default 30)')
parser.add_argument('--no_filters_2', type=int, default=20, metavar='N',
                    help='Number of activation maps to in layer 2 (default 20)')
parser.add_argument('--kernel_0', type=int, default=5, metavar='N',
                    help='Kernel for layer 0 (default 5)')
parser.add_argument('--kernel_1', type=int, default=5, metavar='N',
                    help='Kernel for layer 1 (default 5)')
parser.add_argument('--kernel_2', type=int, default=5, metavar='N',
                    help='Kernel for layer 2 (default 5)')
parser.add_argument('--hidden_0', type=int, default=80, metavar='N',
                    help='Hidden neurons in policy network for layer 0 (default 80)')
parser.add_argument('--hidden_1', type=int, default=60, metavar='N',
                    help='Hidden neurons in policy network for layer 1 (default 60)')
parser.add_argument('--hidden_2', type=int, default=40, metavar='N',
                    help='Hidden neurons in policy network for layer 2 (default 40)')
parser.add_argument('--num_actions', type=int, default=5, metavar='N',
                    help='Number of actions possible for our robot (default 5)')
parser.add_argument('--lr', type=float, default=.0001,
                    help='The learning rate used for my Adam optimizer (default: .0001)')
parser.add_argument('--strides', type=int, default=2, metavar='N',
                    help='Strides for the convolutions in the convlstm layers (default 2)')
parser.add_argument('--drop_rte', type=float, default=.3, metavar='N',
                    help='dropout rate (default .3)')
parser.add_argument('--update_size', type=int, default=30, metavar='N',
                    help='Number of trials a specific policy is run on before we train our models (default 100)')
args = parser.parse_args()

''' Load Deep Models '''
rgb_shape = (3, 64, 64)
dd_inp_shape = (7)
dd_output_shape = (6)
pn_inp_shape = 50 #This is not correct yet

ca_model = Custom_Spatial_Temporal_Anticipation_NN(rgb_shape, (args.no_filters_0,
    args.no_filters_1, args.no_filters_2), (args.kernel_0, args.kernel_0), args.strides, 1,
    padding=0, dropout_rte=args.drop_rte)

dd_model = Deep_Dynamics(dd_inp_shape, 40, 30, 20, 15, 15, dd_output_shape, args.drop_rte)

pn_model = Policy_Network(pn_inp_shape, args.hidden_0, args.hidden_1, args.hidden_2, args.num_actions, args.drop_rte)

if torch.cuda.is_available():
    print("Using GPU acceleration")
    ca_model.cuda()
    dd_model.cuda()
    pn_model.cuda()

ca_optimizer = torch.optim.Adam(ca_model.parameters(), lr=args.lr)
dd_optimizer = torch.optim.Adam(dd_model.parameters(), lr=args.lr)
pn_optimizer = torch.optim.Adam(pn_model.parameters(), lr=args.lr)

def main():
    global dd_model
    global ca_model
    global pn_model
    global ca_optimizer
    global dd_optimizer
    global pn_optimizer
    # first pass of loading and training -- dd so that I can then split the data to its respective owners
    ca_loader = VideoDataGenerator(base_dir + '/data_generated/saved_data/')
    dd_loader = DeepDynamicsDataLoader(base_dir + '/data_generated/current_batch/', base_dir + '/data_generated/saved_data/')
    pn_loader = PolicyNetDataLoader()

    # execute_exp(0, args.update_size) #initial data collection just to train first iteration of dd model
    tr_data, tr_label, val_data, val_label = dd_loader.prepare_first_train()
    train_dd_model(dd_model, dd_optimizer, 100, tr_data, tr_label, val_data, val_label, args.batch_size) #train initial deep dynamics model

    for index in range(args.training_iterations):
        execute_exp(0, args.update_size)
        data, filenames = dd_loader.prepare_last_batch()
        determine_pain_classification(dd_model, data, filenames, base_dir + '/data_generated/', args.num_forward_passes, index)

        #update Anticipation Model --
        tr_data, tr_label, val_data, val_label = ca_loader.prepare_data()
        if len(tr_data) < args.batch_size or len(val_data) < args.batch_size:
            update_anticipation_model(ca_model, ca_optimizer, 10, tr_data, tr_label, val_data, val_label, min([len(tr_data), len(val_data)]))
        else:
            update_anticipation_model(ca_model, ca_optimizer, 10, tr_data, tr_label, val_data, val_label, args.batch_size)

        #update pn network

        if (index + 1) % 10 == 0:
            tr_data, tr_label, val_data, val_label = dd_loader.prepare_data()
            train_dd_model(dd_model, dd_optimizer, 100, tr_data, tr_label, val_data, val_label, args.batch_size)

        #move them into hit and miss category


if __name__ == '__main__':
    main()
