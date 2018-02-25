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
def train_model(model, optimizer, epoch, data, label, batch_size):

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
def validate_model(model, epoch, val_data, val_label, batch_size):
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

    for index in range(int(single_vid.shape[0])):
        lst = []
        input_to_model = torch.from_numpy(single_vid[index, :7])
        if torch.cuda.is_available():
            input_to_model = input_to_model.cuda()
        input_to_model = Variable(input_to_model.float(), volatile=True)
        for inner_index in range(num_forward_passes):
            lst.append((model(input_to_model).cpu().data.numpy()))
        small = calc_statistics(np.asarray(lst), single_vid[index, 7:])
        if smallest > small:
            smallest = small
        del(lst[:])
    return smallest

'''Calculate "probability kinda" of hit'''
def calc_statistics(lst, recorded_state):
    distribution = []
    mean = np.mean(lst)
    var = np.var(lst)
    f = stats.multivariate_normal.pdf(recorded_state, mean=mean, cov=var)
    return np.min(f)

def train_dd_model(model, optimizer, iterations, tr_data, tr_label, val_data, val_label, batch_size):
    for index in range(iterations):
        train_model(model, optimizer, index, tr_data, tr_label, batch_size)
        validate_model(model, index, val_data, val_label, batch_size)

def move_data_files(index_lst, number_corresponds_to_indx, base_dir, iteration):
    #load the file and image
    for index, element in enumerate(index_lst):
        image = np.load(base_dir + 'current_batch/image/' + number_corresponds_to_indx[element])
        state = np.load(base_dir + 'current_batch/state/' + number_corresponds_to_indx[element])
        if (index * 1.0) / (len(index_lst) * 1.0) < .5:

            np.save(base_dir + 'saved_data/miss_image/' + str(element) + 'collision' + str(iteration), image)
            np.save(base_dir + 'saved_data/miss_state/' + str(element) + 'collision' + str(iteration), state)
        else:

            np.save(base_dir + 'saved_data/hit_image/' + str(element) + 'collision' + str(iteration), image)
            np.save(base_dir + 'saved_data/hit_state/' + str(element) + 'collision' + str(iteration), state)


def determine_pain_classification(model, lst, filenames, base_dir, num_forward_passes, iteration):
    pdf_values = []
    index = 0
    for element in lst:
        pdf_values.append(evaluate_model(model, num_forward_passes, element))
        index+=1

    move_data_files((np.asarray(pdf_values)).argsort(), filenames, base_dir, iteration)

    #I need to have a method that saves the images and states into their respective new files
