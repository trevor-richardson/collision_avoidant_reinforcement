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
from scipy import stats

''' Train Model '''
def dd_train_model(model, optimizer, epoch, data, label, batch_size):

    model.train()
    train_loss = 0
    step_counter = 0

    for iteration in range(int(int(data.shape[0])/batch_size)):
        input_to_model = torch.from_numpy(data[(iteration * batch_size):((iteration+1)*batch_size)])
        y_ = torch.from_numpy(label[(iteration * batch_size):((iteration+1)*batch_size)])
        if torch.cuda.is_available():
            input_to_model = input_to_model.cuda()
            y_ = y_.cuda()
        input_to_model = Variable(input_to_model.float())

        y_ = Variable(y_.float())

        pred = model(input_to_model)

        loss = F.mse_loss(pred, y_)

        loss.backward()
        optimizer.step()
        train_loss+=loss.data
        step_counter +=1

    print('Train Deep Dynamics Epoch: {}\tLoss: {:.6f}'.format(
        epoch, train_loss.cpu().numpy()[0]/step_counter))



''' Validate Model '''
def dd_validate_model(model, epoch, val_data, val_label, batch_size):
    model.eval()
    test_loss = 0
    step_counter = 0

    for iteration in range(int(int(val_data.shape[0])/batch_size)):
        input_to_model = torch.from_numpy(val_data[(iteration * batch_size):((iteration+1)*batch_size)])
        y_ = torch.from_numpy(val_label[(iteration * batch_size):((iteration+1)*batch_size)])
        if torch.cuda.is_available():
            input_to_model = input_to_model.cuda()
            y_ = y_.cuda()
        input_to_model = Variable(input_to_model.float(), volatile=True)
        y_ = Variable(y_.float())

        pred = model(input_to_model)

        loss = F.mse_loss(pred, y_)

        test_loss+=loss.data
        step_counter +=1

    print('Test Deep Dynamics Epoch: {}\tLoss: {:.6f}'.format(
        epoch, test_loss.cpu().numpy()[0]/step_counter))

    return test_loss.cpu().numpy()[0]/step_counter

#load 70% of the data for training save best validation error
'''Test what strategy allows me to classify hits vs misses'''
def evaluate_model(model, num_forward_passes, single_vid):
    model.train()
    smallest = 999999999
    rewards = []

    for index in range(int(single_vid.shape[0])):
        lst = []
        input_to_model = torch.from_numpy(single_vid[index, :7])
        if torch.cuda.is_available():
            input_to_model = input_to_model.cuda()
        input_to_model = Variable(input_to_model.float(), volatile=True)
        for inner_index in range(num_forward_passes):
            lst.append((model(input_to_model).cpu().data.numpy()))
        rewards.append(calc_statistics(np.asarray(lst), single_vid[index, 7:]))
        del(lst[:])
    return rewards

'''Calculate "probability kinda" of hit'''
def calc_statistics(lst, recorded_state):
    distribution = []
    mean = np.mean(lst, axis=0)
    var = np.cov(lst, rowvar=False)
    pdf = stats.multivariate_normal.pdf(recorded_state, mean=mean, cov=var)
    return pdf

def train_dd_model(model, optimizer, iterations, tr_data, tr_label, val_data, val_label, batch_size):
    for index in range(iterations):
        dd_train_model(model, optimizer, index, tr_data, tr_label, batch_size)
        dd_validate_model(model, index, val_data, val_label, batch_size)

def move_data_files(index_lst, number_corresponds_to_indx, base_dir, iteration):
    train_pn_lst_hit = []
    train_pn_lst_miss = []

    for index, element in enumerate(index_lst):
        image = np.load(base_dir + 'current_batch/image/' + number_corresponds_to_indx[element])
        state = np.load(base_dir + 'current_batch/state/' + number_corresponds_to_indx[element])

    return train_pn_lst_hit, train_pn_lst_miss

def determine_pain_classification(model, lst, filenames, base_dir, num_forward_passes, iteration):
    pdf_values = []
    index = 0
    for element in lst:
        pdf_values.append(evaluate_model(model, num_forward_passes, element))
        index+=1

    move_data_files(pdf_values, filenames, base_dir, iteration)
    return pdf_values

def determine_reward(dd_model, pn_model, data, num_forward_passes):
    adder = []
    for index in range(data.shape[0] - 1):
        x = np.concatenate((data[index], data[index+1, :6]))
        adder.append(x)
    adder = np.asarray(adder)

    pdf_values = evaluate_model(dd_model, num_forward_passes, adder)
    for element in pdf_values:
        pn_model.rewards.append(element)
