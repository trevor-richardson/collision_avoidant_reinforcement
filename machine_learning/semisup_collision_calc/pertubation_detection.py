import time
import numpy as np
import random
import math
import sys
import scipy.io as sio
import os
from os.path import isfile, join
from os import listdir

import scipy as sp
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
import configparser
from scipy import stats

'''

The following contains functions that are helpful for doing stochastic forward passes on our predictive deep dynamics model
in order to gain a "belief" distribution over the next state. The covariance matrix and the mean of the distribution is used
with exponential smoothin in order to calculate the norm between the actual following state and the smoothed norm.

This distance represents my "pertubation classification"

'''

'''Test what strategy allows me to classify hits vs misses'''
def evaluate_model(model, num_forward_passes, single_vid):
    model.train()
    smallest = 999999999
    rewards = []
    rew = []

    for index in range(int(single_vid.shape[0] -1)):
        lst = []
        input_to_model = torch.from_numpy(single_vid[index])
        if torch.cuda.is_available():
            input_to_model = input_to_model.cuda()
        input_to_model = Variable(input_to_model.float(), volatile=True)
        for inner_index in range(num_forward_passes):
            lst.append((model(input_to_model).cpu().data.numpy()))
        n1, n2 = calc_norm_2(np.asarray(lst), single_vid[index + 1, :12])
        rew.append(n2)
        del(lst[:])
    return rew


'''Calculate "probability kinda" of hit'''
def calc_statistics(lst, recorded_state):
    distribution = []
    mean = np.mean(lst, axis=0)
    covar = np.cov(lst, rowvar=False)
    pdf = stats.multivariate_normal.pdf(recorded_state, mean=mean, cov=covar)
    return pdf

def calc_norm_1(lst, recorded_state, mean):
    delta = mean - recorded_state
    return delta, np.linalg.norm(delta)

'''Norm 2 works better because of the inclusion of exponential smoothing and the covariance relationship'''
def calc_norm_2(lst, recorded_state):
    mean = np.mean(lst, axis=0)
    covar = np.cov(lst, rowvar=False)
    delta, norm_delta = calc_norm_1(lst, recorded_state, mean)
    delta_2 = np.exp(delta - 2*covar)
    return norm_delta, np.linalg.norm(delta_2)

def calc_confidence_interval(data, confidence=0.90):
    n = len(data)
    m, se = np.mean(data), scipy.stats.sem(data)

    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m-h, m+h

def determine_reward(dd_model, pn_model, data, num_forward_passes):

    rew = evaluate_model(dd_model, num_forward_passes, data)
    low, high = calc_confidence_interval(rew)
    minimum = min(rew)

    for i in range(len(rew)):
        if rew[i] < high:
            rew[i] = 0
        else:
            rew[i] += -minimum
            if rew[i] < 5:
                rew[i] = 0
            else:
                rew[i] = 1

    if int(len(pn_model.reset_locations)) > 1:
        size_split = int(int(len(rew)) / (pn_model.reset_locations[-1] - pn_model.reset_locations[-2]))
        num_splits = pn_model.reset_locations[-1] - pn_model.reset_locations[-2]
    else:
        size_split = int(int(len(rew)) / int(pn_model.reset_locations[-1] + 1))
        num_splits = pn_model.reset_locations[0] + 1

    for indx in range(num_splits):
        if indx + 1 == num_splits:
            pn_model.rewards.append(-max(rew[indx*size_split:]))
        else:
            pn_model.rewards.append(-max(rew[indx*size_split:((indx+1) * size_split)]))


def determine_reward_val(dd_model, pn_model, data, num_forward_passes):

    rew = evaluate_model(dd_model, num_forward_passes, data)
    low, high = calc_confidence_interval(rew)
    minimum = min(rew)

    for i in range(len(rew)):
        if rew[i] < high:
            rew[i] = 0
        else:
            rew[i] += -minimum
            if rew[i] < 5:
                rew[i] = 0
            else:
                return 1
    return 0


def dd_test(dd_model, num_forward_passes, data):
    rew = evaluate_model(dd_model, num_forward_passes, data)
    return rew


def determine_reward_no_repeat(dd_model, pn_model, data, num_forward_passes, only_hits, pos_for_miss):
    rew = evaluate_model(dd_model, num_forward_passes, data)
    low, high = calc_confidence_interval(rew)
    minimum = min(rew)

    for i in range(len(rew)):
        if rew[i] < high:
            rew[i] = 0
        else:
            rew[i] += -minimum
            if rew[i] < 5:
                rew[i] = 0
            else:
                rew[i] = -1

    if pos_for_miss:
        if min(rew) == 0:
            for i in range(len(rew)):
                rew[i] = 1
        else:
            for i in range(len(rew)):
                rew[i] = -1

    if only_hits:
        if min(rew) == -1:
            for index, element in enumerate(rew):
                pn_model.rewards.append(element)
                pn_model.saved_log_probs.append(pn_model.current_log_probs[index])
            pn_model.saved_log_probs.append(pn_model.current_log_probs[-1])
            pn_model.rewards.append(rew[-1])
            pn_model.reset_locations.append(len(pn_model.saved_log_probs) -1)
        else:
            for element in pn_model.current_log_probs:
                element.detach_()
    else:
        for index, element in enumerate(rew):
            pn_model.rewards.append(element)
            pn_model.saved_log_probs.append(pn_model.current_log_probs[index])
        pn_model.saved_log_probs.append(pn_model.current_log_probs[-1])
        pn_model.rewards.append(rew[-1])
        pn_model.reset_locations.append(len(pn_model.saved_log_probs) -1)

    pn_model.current_log_probs = []
