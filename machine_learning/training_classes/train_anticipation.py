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

'''Training'''
def train_model(model, optimizer, epoch, data_files, label, batch_size):
    model.train()

    tim = time.time()
    predicted_list = []
    y_list = []
    train_loss = 0
    train_accuracy = 0
    train_step_counter = 0

    for index in range(int(len(data_files)/batch_size)):
        current_video = load_next_batch(data_files[index*batch_size:(index+1)*batch_size])
        current_label = np.asarray(label[index*batch_size:(index+1)*batch_size])
        target = torch.from_numpy((current_label))
        if torch.cuda.is_available():
            target = target.cuda()
        target = Variable(target)

        prev0 = create_lstm_states(model.convlstm_0.output_shape, batch_size)
        prev1 = create_lstm_states(model.convlstm_1.output_shape, batch_size)
        prev2 = create_lstm_states(model.convlstm_2.output_shape, batch_size)
        states = [prev0, prev1, prev2]
        optimizer.zero_grad()

        for inner_index in range(int(current_video.shape[0])):
            data = torch.from_numpy(current_video[inner_index]).float()
            if torch.cuda.is_available():
                data = data.cuda()
            data = Variable(data)

            output, states = model(data, states)
            predicted_list.append(output)
            y_list.append(target)

        pred = torch.cat(predicted_list)
        y_ = torch.cat(y_list).float()
        loss = F.binary_cross_entropy(pred, y_)

        loss.backward()
        optimizer.step()
        train_loss+=loss.data
        train_step_counter +=1

        del(predicted_list[:])
        del(y_list[:])

    print("Training time for one epoch", time.time() - tim)
    print('Train Epoch: {}\tLoss: {:.6f}'.format(
        epoch, train_loss.cpu().numpy()[0]/train_step_counter))


'''Testing/Validation'''
def test_model(model, optimizer, data_files, label, batch_size):
    model.eval()

    tim = time.time()
    test_loss = 0
    correct = 0
    instance_counter = 0
    test_step_counter = 0

    for index in range(int(len(data_files)/batch_size)):
        current_video = load_next_batch(data_files[index*batch_size:(index+1)*batch_size])
        current_label = np.asarray(label[index*batch_size:(index+1)*batch_size])
        target = torch.from_numpy((current_label))
        if torch.cuda.is_available():
            target = target.cuda()
        target = Variable(target.float())

        prev0 = create_lstm_states(model.convlstm_0.output_shape, batch_size)
        prev1 = create_lstm_states(model.convlstm_1.output_shape, batch_size)
        prev2 = create_lstm_states(model.convlstm_2.output_shape, batch_size)
        states = [prev0, prev1, prev2]

        for inner_index in range(int(current_video.shape[0])):
            data = torch.from_numpy(current_video[inner_index]).float()
            if torch.cuda.is_available():
                data = data.cuda()
            data = Variable(data, volatile=True)

            output, states = model(data, states)
            test_loss += F.binary_cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
            pred = torch.round(output.data) # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum()
            instance_counter+=1


    test_loss /= (instance_counter * batch_size)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss, correct, instance_counter * batch_size,
        100. * correct / (instance_counter * batch_size)))
    return 100. * correct / (instance_counter * batch_size)


'''Visualize Activations'''

def visualize_learning(data_files, label, view_hit, h_or_c, batch_size):
    model.eval()
    index = 0
    if view_hit:
        for element in label:
            if element == 1:
                break
            index+=1
    else:
        for element in label:
            if element == 0:
                break
            index+=1

    tim = time.time()
    test_loss = 0
    correct = 0
    instance_counter = 0
    test_step_counter = 0
    activation_list = []

    current_video = load_next_batch(data_files[index:(index+1)])
    current_label = np.asarray(label[index:(index+1)])

    target = torch.from_numpy((current_label))

    if torch.cuda.is_available():
        target = target.cuda()
    target = Variable(target.float())

    prev0 = create_lstm_states(model.convlstm_0.output_shape, 1)
    prev1 = create_lstm_states(model.convlstm_1.output_shape, 1)
    prev2 = create_lstm_states(model.convlstm_2.output_shape, 1)
    states = [prev0, prev1, prev2]

    for inner_index in range(int(current_video.shape[0])):
        data = torch.from_numpy(current_video[inner_index]).float()
        if torch.cuda.is_available():
            data = data.cuda()
        data = Variable(data, volatile=True)

        output, states = model(data, states)

        visualize_0 = np.transpose(np.asarray([states[0][0].data.cpu().numpy(), states[0][1].data.cpu().numpy()]),(1, 0, 3, 4, 2))
        visualize_1 = np.transpose(np.asarray([states[1][0].data.cpu().numpy(), states[1][1].data.cpu().numpy()]),(1, 0, 3, 4, 2))
        visualize_2 = np.transpose(np.asarray([states[2][0].data.cpu().numpy(), states[2][1].data.cpu().numpy()]),(1, 0, 3, 4, 2))

        activation_list.append([visualize_0, visualize_1, visualize_2])

        test_loss += F.binary_cross_entropy(output, target, size_average=False).data[0] # sum up batch loss
        pred = torch.round(output.data) # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).sum()
        instance_counter+=1

    visualizer = VisualizeActivations(activation_list, np.squeeze(current_video), h_or_c)
    visualizer.visualize_activation()


    test_loss /= (instance_counter * batch_size)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss, correct, instance_counter * batch_size,
        100. * correct / (instance_counter * batch_size)))
    return 100. * correct / (instance_counter * batch_size)


def update_anticipation_model(model, optimizer, iterations, tr_data, tr_label, val_data, val_label, batch_size):
    for iteration in range(iterations):
        train_model(model, optimizer, iteration, tr_data, tr_label, batch_size)
        test_model(model, optimizer, val_data, val_label, batch_size)


'''Helper Functions'''
def create_lstm_states(shape, batch_size):
    c = Variable(torch.zeros(batch_size, shape[0], shape[1], shape[2])).float().cuda()
    h = Variable(torch.zeros(batch_size, shape[0], shape[1], shape[2])).float().cuda()
    return (h, c)


def load_next_batch(string_names):
    lst = []
    for element in string_names:
    #list of string names
        data = np.load(element)
        lst.append(data)
    movies = np.stack(lst, axis =0)
    movies = np.transpose(movies, (1, 0, 2, 3, 4))
    return movies


def print_parameters(params_list):
    total = 0
    for element in params_list:
        adder = 1
        for inner_element in element.size():
            adder*=inner_element
        print(element.size())
        total+=adder
    print("Total number of paramters in model:", total)


def view_image(image, name):
    plt.imshow(image.numpy(), cmap='gray')
    plt.title(name)
    plt.show()
