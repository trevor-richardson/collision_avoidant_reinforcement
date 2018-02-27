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

        loss = F.nll_loss(pred, y_)

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
