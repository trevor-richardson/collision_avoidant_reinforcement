import numpy as np
import random
import math
import scipy.io as sio
from os.path import isfile, join
from os import listdir

class DeepDynamicsDataLoader(object):
    def __init__(self, current_dir, memory_dir):
        print("Loading Deep Dynamics Data Loader")

        self.current_dir = current_dir + 'state/'
        self.memory_dir = memory_dir

    def prepare_last_batch(self):
        current = [f for f in listdir(self.current_dir) if isfile(join(self.current_dir, f))]
        returned = []
        for element in current:
            returned.append(element)

        for indx, element in enumerate(current):
            current[indx] = self.current_dir + element

        with_label = []
        for element in current:
            arr = np.load(element)
            adder = []
            for index in range(arr.shape[0] - 1):
                x = np.concatenate((arr[index], arr[index+1, :6]))
                adder.append(x)
            with_label.append(np.round(np.asarray(adder), 2))

        return with_label, returned

    def prepare_first_train(self):
        current = [f for f in listdir(self.current_dir) if isfile(join(self.current_dir, f))]

        for indx, element in enumerate(current):
            current[indx] = self.current_dir + element

        with_label = []
        for element in current:
            arr = np.load(element)
            adder = []
            for index in range(arr.shape[0] - 1):
                x = np.concatenate((arr[index], arr[index+1, :6]))
                adder.append(x)
            with_label.append(np.asarray(adder))

        data = with_label[0]
        for index in range(int(len(with_label) *.7)):
            data = np.concatenate((data, with_label[index+1]), axis=0)

        np.random.shuffle(data)

        train_data = data[:, :7]
        train_label = data[:, 7:]

        train_data = np.round(train_data, 2)
        train_label = np.round(train_label, 2)

        #get basic stats about data
        val_data = train_data[int(.7 * train_data.shape[0]):]
        val_label = train_label[int(.7 * train_data.shape[0]):]
        data = train_data[:int(.7 * train_data.shape[0])]
        label = train_label[:int(.7 * train_data.shape[0])]

        return data, label, val_data, val_label


    def prepare_data(self):
        hit_dir = self.memory_dir + '/hit_state/'
        miss_dir = self.memory_dir + '/miss_state/'

        ground_truth = []

        hit = [f for f in listdir(hit_dir) if isfile(join(hit_dir, f))]
        miss = [f for f in listdir(miss_dir) if isfile(join(miss_dir, f))]

        for indx, element in enumerate(hit):
            hit[indx] = hit_dir + element
        for indx, element in enumerate(miss):
            miss[indx] = miss_dir + element

        #here is where the split needs to occur
        hit_with_label = []
        for element in hit:
            arr = np.load(element)
            adder = []
            for index in range(arr.shape[0] - 1):
                x = np.concatenate((arr[index], arr[index+1, :6]))
                adder.append(x)
            hit_with_label.append(np.asarray(adder))

        miss_with_label = []
        for element in miss:
            arr = np.load(element)
            adder = []
            for index in range(arr.shape[0] - 1):
                x = np.concatenate((arr[index], arr[index+1, :6]))
                adder.append(x)
            miss_with_label.append(np.asarray(adder))

        data = hit_with_label[0]
        for index in range(int(len(hit_with_label) *.7)):
            data = np.concatenate((data, hit_with_label[index+1]), axis=0)

        for index in range(int(len(miss_with_label) * .7)):
            data = np.concatenate((data, miss_with_label[index]), axis=0)

        np.random.shuffle(data)

        train_data = data[int(.8 * int(data.shape[0])):, :7]
        train_label = data[int(.8 * int(data.shape[0])):, 7:]

        val_data = data[:int(.8 * int(data.shape[0])), :7]
        val_label = data[:int(.8 * int(data.shape[0])), 7:]

        return train_data, train_label, val_data, val_label
