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
from run_vrep_simulation import execute_exp
from train_dd import *
from train_anticipation import *

'''
I need some way to save the models together -- every 100 steps I want to retrain Anticipation model and deep dynamics
I need to be able to deal with a batch of data - x
I need to be able to decide when I've converged on training dd and Collision Anticipation
Update collision ancitipation model
----------------------
Save old demos and their results --
Update new deep dynamics model --
Update collision prediction system --
Run new experiments
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


parser = argparse.ArgumentParser(description='Reinforcement Learning Guided by Deep Dynamics and Anticipation Models in Pytorch')

#training and testing args
parser.add_argument('--training_iterations', type=int, default=30000, metavar='N',
                    help='Number of times I want to train a reinforcement learning model')
parser.add_argument('--num_forward_passes', type=int, default=32, metavar='N',
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
                    help='Hidden neurons in policy network for layer 0 (default 100)')
parser.add_argument('--hidden_1', type=int, default=60, metavar='N',
                    help='Hidden neurons in policy network for layer 1 (default 60)')
parser.add_argument('--hidden_2', type=int, default=40, metavar='N',
                    help='Hidden neurons in policy network for layer 2 (default 40)')
parser.add_argument('--hidden_3', type=int, default=40, metavar='N',
                    help='Hidden neurons in policy network for layer 2 (default 20)')
parser.add_argument('--num_actions', type=int, default=5, metavar='N',
                    help='Number of actions possible for our robot (default 5)')
parser.add_argument('--lr', type=float, default=.0001,
                    help='The learning rate used for my Adam optimizer (default: .0001)')
parser.add_argument('--strides', type=int, default=2, metavar='N',
                    help='Strides for the convolutions in the convlstm layers (default 2)')
parser.add_argument('--drop_rte', type=float, default=.3, metavar='N',
                    help='dropout rate (default .3)')
parser.add_argument('--gamma', type=float, default=.99, metavar='G',
                    help='discount factor (default: 0.999)')
parser.add_argument('--update_size', type=int, default=100, metavar='N',
                    help='Number of trials a specific policy is run on before we train our models (default 100)')
args = parser.parse_args()

''' Load Deep Models '''
rgb_shape = (3, 64, 64)
dd_inp_shape = (7)
dd_output_shape = (6)
pn_inp_shape = 3 * 64 * 64 + 7  #Input is a flattened image and the current state

ca_model = Custom_Spatial_Temporal_Anticipation_NN(rgb_shape, (args.no_filters_0,
    args.no_filters_1, args.no_filters_2), (args.kernel_0, args.kernel_0), args.strides, 1,
    padding=0, dropout_rte=args.drop_rte)

dd_model = Deep_Dynamics(dd_inp_shape, 40, 30, 20, 15, 15, dd_output_shape, args.drop_rte)

pn_model = Policy_Network(pn_inp_shape, args.hidden_0, args.hidden_1, args.hidden_2, args.hidden_3, args.num_actions)

if torch.cuda.is_available():
    print("Using GPU acceleration")
    ca_model.cuda()
    dd_model.cuda()
    pn_model.cuda()

ca_optimizer = torch.optim.Adam(ca_model.parameters(), lr=args.lr)
dd_optimizer = torch.optim.Adam(dd_model.parameters(), lr=args.lr)
pn_optimizer = torch.optim.Adam(pn_model.parameters(), lr=1e-2)

def save_models(iteration):
    torch.save(ca_model.state_dict(), base_dir + '/machine_learning/saved_models/ca' + str(iteration) + '.pth')
    torch.save(pn_model.state_dict(), base_dir + '/machine_learning/saved_models/pn' + str(iteration) + '.pth')
    torch.save(dd_model.state_dict(), base_dir + '/machine_learning/saved_models/dd' + str(iteration) + '.pth')

def load_models(iteration):
    global dd_model
    global pn_model
    global ca_model
    try:
        dd_model.load_state_dict(torch.load(base_dir + "/machine_learning/saved_models/dd" + str(iteration) + ".pth"))
        pn_model.load_state_dict(torch.load(base_dir + "/machine_learning/saved_models/pn" + str(iteration) + ".pth"))
        ca_model.load_state_dict(torch.load(base_dir + "/machine_learning/saved_models/ca" + str(iteration) + ".pth"))
    except ValueError:
        print("Not a valid model to load")
        sys.exit()

def update_policy_network(model, optimizer):
    R = 0
    policy_loss = []
    rewards = []
    count = 0
    counter = len(model.saved_log_probs) - 1
    count_reset = len(model.reset_locations) - 1
    for r in model.rewards[::-1]:
        if model.reset_locations[count_reset] == counter:
            R = 0
            count_reset = count_reset -1
        R = r + args.gamma * R
        rewards.insert(0, R)
        counter = counter -1
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    for log_prob, reward in zip(model.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.reset_locations[:]
    del model.saved_log_probs[:]
    del model.updated_log_probs[:]
    print(rewards.sum())
    return R

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

    #need to make policy network work in simulation
    # load_models(498)
    # execute_exp(pn_model, 0, args.update_size, False) #initial data collection just to train first iteration of dd model
    tr_data, tr_label, val_data, val_label = dd_loader.prepare_first_train()
    train_dd_model(dd_model, dd_optimizer, 50, tr_data, tr_label, val_data, val_label, args.batch_size) #train initial deep dynamics model
    pn_optimizer.zero_grad()
    print("####################################################################################################################\n")
    for index in range(args.training_iterations):

        data = execute_exp(pn_model, 0, 1, True) #needs to return batch, necesary_arguments,
        pn_model.saved_log_probs = pn_model.saved_log_probs[:-1]
        pn_model.saved_log_probs = pn_model.saved_log_probs[:-1]
        determine_reward(dd_model, pn_model, data[0][1], args.num_forward_passes)
        if len(pn_model.reset_locations) > 1:
            pn_model.reset_locations[-1] += pn_model.reset_locations[-2] + 1

        if (index + 1) % 5 == 0:
            reward = update_policy_network(pn_model, pn_optimizer)
            print("Reward", reward)
            print()
            print("####################################################################################################################\n")

        if (index + 3) % 50 == 0:
            pn_optimizer.zero_grad()
            save_models(index)
            # data = execute_exp(pn_model, args.update_size + 100, args.update_size + 100 + args.update_size, False)
            # tr_data, tr_label, val_data, val_label = dd_loader.prepare_first_train()
            # train_dd_model(dd_model, dd_optimizer, 75, tr_data, tr_label, val_data, val_label, args.batch_size)




if __name__ == '__main__':
    main()
