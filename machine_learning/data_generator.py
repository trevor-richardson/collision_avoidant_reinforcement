import numpy as np
import random
import math
import scipy.io as sio
from os.path import isfile, join
from os import listdir

class VideoDataGenerator(object):
    def __init__(self, train_dir, val_dir, test_dir, train, val, test):
        self.training_directory = train_dir
        self.validation_directory = val_dir
        self.testing_directory = test_dir
        self.train_number_of_hit = train
        self.train_number_of_miss = train
        self.val_number_of_hit = val
        self.val_number_of_miss = val
        self.test_number_of_hit = test
        self.test_number_of_miss = test

    def prepare_data(self):
        print("Preparing data files")
        self.train = []
        self.val = []
        self.test = []
        self.train_class = []
        self.val_class = []
        self.test_class = []

        classification1 = np.array([1])
        classification0 = np.array([0])

        train_hit_dir = self.training_directory + 'hit/'
        train_miss_dir = self.training_directory + 'miss/'
        val_hit_dir = self.validation_directory + 'hit/'
        val_miss_dir = self.validation_directory + 'miss/'
        test_hit_dir = self.testing_directory + 'hit/'
        test_miss_dir = self.testing_directory + 'miss/'

        train_hit = []
        train_miss = []
        val_hit = []
        val_miss = []
        test_hit = []
        test_miss = []

        train_hit = [f for f in listdir(train_hit_dir) if isfile(join(train_hit_dir, f))]
        train_miss = [f for f in listdir(train_miss_dir) if isfile(join(train_miss_dir, f))]
        test_hit= [f for f in listdir(test_hit_dir) if isfile(join(test_hit_dir, f))]
        test_miss = [f for f in listdir(test_miss_dir) if isfile(join(test_miss_dir, f))]
        val_hit = [f for f in listdir(val_hit_dir) if isfile(join(val_hit_dir, f))]
        val_miss = [f for f in listdir(val_miss_dir) if isfile(join(val_miss_dir, f))]

        for indx, element in enumerate(train_hit):
            train_hit[indx] = train_hit_dir + element
        for indx, element in enumerate(train_miss):
            train_miss[indx] = train_miss_dir + element
        for indx, element in enumerate(val_hit):
            val_hit[indx] = val_hit_dir + element
        for indx, element in enumerate(val_miss):
            val_miss[indx] = val_miss_dir + element
        for indx, element in enumerate(test_hit):
            test_hit[indx] = test_hit_dir + element
        for indx, element in enumerate(test_miss):
            test_miss[indx] = test_miss_dir + element

        #shuffle the list of data names for train and test individually
        hit_counter = 0
        miss_counter = 0
        sizeoftrain = self.train_number_of_hit + self.train_number_of_miss
        for enumerator in range(sizeoftrain):
            rand = random.uniform(0,1)
            if (miss_counter > self.train_number_of_hit -1):
                self.train.append(train_hit[hit_counter])
                self.train_class.append(classification1)
                hit_counter+=1
            elif (hit_counter > self.train_number_of_miss -1):
                self.train.append(train_miss[miss_counter])
                self.train_class.append(classification0)
                miss_counter+=1
            elif (rand < .5):
                self.train.append(train_hit[hit_counter])
                self.train_class.append(classification1)
                hit_counter+=1
            else:
                self.train.append(train_miss[miss_counter])
                self.train_class.append(classification0)
                miss_counter+=1

        hit_counter = 0
        miss_counter = 0
        sizeofval = self.val_number_of_hit + self.val_number_of_miss
        for enumerator in range(sizeofval):
            rand = random.uniform(0,1)
            if (miss_counter > self.val_number_of_hit -1):
                self.val.append(val_hit[hit_counter])
                self.val_class.append(classification1)
                hit_counter+=1
            elif (hit_counter > self.val_number_of_miss -1):
                self.val.append(val_miss[miss_counter])
                self.val_class.append(classification0)
                miss_counter+=1
            elif (rand < .5):
                self.val.append(val_hit[hit_counter])
                self.val_class.append(classification1)
                hit_counter+=1
            else:
                self.val.append(val_miss[miss_counter])
                self.val_class.append(classification0)
                miss_counter+=1

        hit_counter = 0
        miss_counter = 0
        sizeoftest = self.test_number_of_hit + self.test_number_of_miss
        for enumerator in range(sizeoftest):
            rand = random.uniform(0,1)
            if (miss_counter > self.test_number_of_miss -1):
                self.test.append(test_hit[hit_counter])
                self.test_class.append(classification1)
                hit_counter+=1
            elif (hit_counter > self.test_number_of_hit -1):
                self.test.append(test_miss[miss_counter])
                self.test_class.append(classification0)
                miss_counter+=1
            elif (rand < .5):
                self.test.append(test_hit[hit_counter])
                self.test_class.append(classification1)
                hit_counter+=1
            else:
                self.test.append(test_miss[miss_counter])
                self.test_class.append(classification0)
                miss_counter+=1
        '''
        Print some facts about my dataset
        '''
        train_hit_count = 0
        test_hit_count = 0
        val_hit_count = 0

        for x in self.train_class:
            if x == classification1:
                train_hit_count+=1
        for y in self.test_class:
            if y == classification1:
                test_hit_count+=1
        for z in self.val_class:
            if z == classification1:
                val_hit_count+=1


        print ("\n\n******************* Creating a Data Generator ************************\n")
        print ("\t\t  Number of train hits : ", train_hit_count)
        print ("\t\t Number of train misses : ", len(self.train_class) - train_hit_count)

        print ("\t\t   Number of val hits : ", val_hit_count)
        print ("\t\t  Number of val misses : ", len(self.val_class) - val_hit_count)

        print ("\t\t  Number of test hits : ", test_hit_count)
        print ("\t\t Number of test misses : ", len(self.test_class) - test_hit_count)
        print ("\n***********************************************************************\n\n")

        return self.train, self.train_class, self.val, self.val_class, self.test, self.test_class
