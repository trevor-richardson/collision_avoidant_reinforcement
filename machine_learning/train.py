'''Train Reinforcement Learning Policy Network
that Learns to Avoid External Pertubation "Pain" '''
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
from policy_network import Policy_Network
from collision_avoidance import AnticipationNet
from policy_convlstm_net import ConvLSTMPolicyNet
from policy_conv_net import ConvPolicy_Network
from policy_lstm_network import Policy_LSTMNetwork
from policy_conv_irnn import ConviRNNPolicy_Network
from train_models_run_vrep import execute_exp
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

''' Global Variables of Interest For Configurability '''

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Reinforcement Learning Guided by Deep Dynamics and Anticipation Models in Pytorch')

#training and testing args
parser.add_argument('--pred_window', type=int, default=10, metavar='N',
                    help='How far in the future collision anticipation can guess')
parser.add_argument('--policy_inp_type', type=int, default=2, metavar='N',
                    help='Type of input for policy net')
parser.add_argument('--training_iterations', type=int, default=30000, metavar='N',
                    help='Number of times I want to train a reinforcement learning model')
parser.add_argument('--num_forward_passes', type=int, default=64, metavar='N',
                    help='Number of forward passes for dropout at test time for multivariate_normal pdf calc')
parser.add_argument('--exp_iteration', type=int, default=64, metavar='N',
                    help='Batch size default size is 64')
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
parser.add_argument('--gamma', type=float, default=.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--update_size', type=int, default=32, metavar='N',
                    help='Number of trials a specific policy is run on before we train our models (default 100)')
parser.add_argument('--use_ca', type=str2bool, nargs='?', default=True,
                    help='Whether or not to use pretrained collision anticpation model')
parser.add_argument('--all_hit', type=str2bool, nargs='?', default=True,
                    help='Whether or not I should use every simulation for training or just hit simulations')
parser.add_argument('--exp_num', type=int, default=0, metavar='N',
                    help='This is to seperate models saved and the results from training')
args = parser.parse_args()

rgb_shape = (3, 64, 64)
dd_inp_shape = (13)
dd_output_shape = (12)
ca_output_shape = 10
h_0 = 15
h_1 = 15
h_2 = 15
h_out = 50
pn_output = 5
eps = np.finfo(np.float32).eps.item()

'''
0 - fcn
1 - conv lstm 2 branches for state action and image
2 - conv network
3 - conv irnn
'''

if args.use_ca:
    if args.policy_inp_type == 0:
        print("Initializing Feed foward policy network")
        pn_inp = 3 * 64 * 64 * 2 + 10 + 13
        pn_model = Policy_Network(pn_inp, args.hidden_0, args.hidden_1, args.hidden_2, args.hidden_3, pn_output)
    elif args.policy_inp_type == 1:
        print("Initializing ConvLSTM Policy Net")
        st_shp = (dd_inp_shape+args.pred_window)
        pn_model = ConvLSTMPolicyNet(rgb_shape, st_shp, h_0, h_1, h_2, h_out, (args.no_filters_0,
            args.no_filters_1, args.no_filters_2), (args.kernel_0, args.kernel_0), args.strides, pn_output,
            padding=0)
    elif args.policy_inp_type == 2:
        print("Initializing Conv Policy Net")
        st_shp = (dd_inp_shape+args.pred_window)
        pn_model = ConvPolicy_Network(st_shp, (6, 64, 64), args.no_filters_0, args.no_filters_1,
            args.no_filters_2, 5, args.hidden_0, args.hidden_1, args.hidden_2, pn_output)
    elif args.policy_inp_type == 3:
        print("Initializing conv-irnn policy")
        st_shp = (dd_inp_shape+args.pred_window)
        pn_model = ConviRNNPolicy_Network(st_shp, (6, 64, 64), args.no_filters_0, args.no_filters_1,
            args.no_filters_2, 5, args.hidden_0, args.hidden_1, pn_output)
        pn_model.reset(1) #reset hidden states of model
    else:
        print("Enter a correct input type")
        sys.exit()
else:
    if args.policy_inp_type == 0:
        print("Initializing feed forward policy")
        pn_inp = 3 * 64 * 64 * 2 + 13
        pn_model = Policy_Network(pn_inp, args.hidden_0, args.hidden_1, args.hidden_2, args.hidden_3, pn_output)
    elif args.policy_inp_type == 1:
        print("Initializing convLSTM policy")
        st_shp = (dd_inp_shape)
        pn_model = ConvLSTMPolicyNet(rgb_shape, st_shp, h_0, h_1, h_2, h_out, (args.no_filters_0,
            args.no_filters_1, args.no_filters_2), (args.kernel_0, args.kernel_0), args.strides, pn_output,
            padding=0)
    elif args.policy_inp_type == 2:
        print("Initializing conv policy")
        st_shp = (dd_inp_shape)
        pn_model = ConvPolicy_Network(st_shp, (6, 64, 64), args.no_filters_0, args.no_filters_1,
            args.no_filters_2, 5, args.hidden_0, args.hidden_1, args.hidden_2, pn_output)
    elif args.policy_inp_type == 3:
        print("Initializing conv-irnn policy")
        st_shp = (dd_inp_shape)
        pn_model = ConviRNNPolicy_Network(st_shp, (6, 64, 64), args.no_filters_0, args.no_filters_1,
            args.no_filters_2, 5, args.hidden_0, args.hidden_1, pn_output)
        pn_model.reset(1) #reset hidden states of model
    else:
        print("Enter a correct input type")
        sys.exit()

ca_model = AnticipationNet(rgb_shape, dd_inp_shape-3, h_0, h_1, h_2, h_out, (args.no_filters_0,
    args.no_filters_1, args.no_filters_2), (args.kernel_0, args.kernel_0), args.strides, args.pred_window,
    padding=0)

dd_model = Deep_Dynamics(dd_inp_shape, 60, 40, 30, 20, 20, dd_output_shape, 0, .6)

if torch.cuda.is_available():
    print("Using GPU acceleration")
    pn_model.cuda()
    ca_model.cuda()
    dd_model.cuda()

ca_optimizer = torch.optim.Adam(ca_model.parameters(), lr=args.lr)
dd_optimizer = torch.optim.Adam(dd_model.parameters(), lr=args.lr)
pn_optimizer = torch.optim.Adam(pn_model.parameters(), lr=args.lr)

def save_models(iteration):
    torch.save(pn_model.state_dict(), base_dir + '/machine_learning/saved_models/' + str(args.exp_num) + 'pn' + str(iteration) + '.pth')

def load_models(iteration):
    global dd_model
    global ca_model
    global pn_model
    try:
        dd_model.load_state_dict(torch.load(base_dir + "/machine_learning/saved_models/dd_model/0.20046165377594705.pth"))
        ca_model.load_state_dict(torch.load(base_dir + "/machine_learning/saved_models/ca_model/780.5778702075141.pth"))
        pn_model.load_state_dict(torch.load(base_dir + "/machine_learning/saved_models/pn" + str(iteration) + ".pth"))
    except ValueError:
        print("Not a valid model to load")
        sys.exit()

def load_dd_model():
    global dd_model
    try:
        dd_model.load_state_dict(torch.load(base_dir + "/machine_learning/saved_models/dd_model/0.25559931709652856.pth"))
    except ValueError:
        print("Not a valid model to load")
        sys.exit()

def load_ca_model():
    global ca_model
    try:
        ca_model.load_state_dict(torch.load(base_dir + "/machine_learning/saved_models/ca_model/780.5778702075141.pth"))
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

    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for log_prob, reward in zip(model.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()  #normalize or scale gradient by total steps
    policy_loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.reset_locations[:]
    del model.saved_log_probs[:]
    del model.current_log_probs[:]
    optimizer.zero_grad()
    return rewards.sum()

load_ca_model()
load_dd_model()
# load_models(3200)

def main():
    global dd_model
    global ca_model
    global pn_model
    global ca_optimizer
    global dd_optimizer
    global pn_optimizer
    num_updates = 0
    update_counter = 1
    index = 0
    count = 0
    count_list = []

    '''The following Collision Anticipation Network is a
        mentor for the Policy Network. It is pretrained, and no grads required'''

    ca_optimizer.zero_grad()
    for param in ca_model.parameters():
        param.requires_grad=False

    tim = time.time()

    while index < args.training_iterations:

        data = execute_exp(ca_model, pn_model, 0, 1, args.policy_inp_type, args.use_ca) #needs to return batch, necesary_arguments,
        if args.policy_inp_type == 3:
            pn_model.reset(1)
        determine_reward_no_repeat(dd_model, pn_model, data[0], args.num_forward_passes, args.all_hit)
        dd_optimizer.zero_grad()
        ca_optimizer.zero_grad()

        if len(pn_model.reset_locations) == args.update_size:

            update_counter+=1
            reward = update_policy_network(pn_model, pn_optimizer)
            index +=1

            pn_optimizer.zero_grad()
            pn_optimizer.zero_grad()
            save_models(index)
            update_counter = 1

            if args.all_hit:
                count_list.append(count)
            else:
                count_list.append(reward)

            print(str(args.exp_num) + "**********************", index ,"**********************")
            print("Iterations from last update or total reward ", count_list)

            count = 0
        count+=1
    np.save(str(args.exp_num) + "tbtwnupdate", np.asarray(count_list))
    print("TIME IT TOOK ", time.time() - tim)

if __name__ == '__main__':
    main()
