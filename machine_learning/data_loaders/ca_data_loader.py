import numpy as np
import random
import math
import scipy.io as sio
from os.path import isfile, join
from os import listdir

'''Returns lists of where the data is stored aka lightweight not holding all of the video data'''

class VideoDataGenerator(object):
    def __init__(self, train_dir):
        self.training_directory = train_dir

    def prepare_data(self):
        print("Preparing data files")
        self.train = []
        self.train_class = []
        self.val = []
        self.val_class = []

        classification1 = np.array([1])
        classification0 = np.array([0])

        train_hit_dir = self.training_directory + 'hit_image/'
        train_miss_dir = self.training_directory + 'miss_image/'

        train_hit = []
        train_miss = []
        val_hit = []
        val_miss = []

        train_hit = [f for f in listdir(train_hit_dir) if isfile(join(train_hit_dir, f))]
        train_miss = [f for f in listdir(train_miss_dir) if isfile(join(train_miss_dir, f))]

        for indx, element in enumerate(train_hit):
            train_hit[indx] = train_hit_dir + element
        for indx, element in enumerate(train_miss):
            train_miss[indx] = train_miss_dir + element

        hit_counter = 0
        miss_counter = 0

        sizeoftrain = train_number_of_hit + train_number_of_miss
        for enumerator in range(sizeoftrain):
            rand = random.uniform(0,1)
            if (miss_counter > train_number_of_hit -1):
                self.train.append(train_hit[hit_counter])
                self.train_class.append(classification1)
                hit_counter+=1
            elif (hit_counter > train_number_of_miss -1):
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

        '''Slice train and provide final train and val lists'''
        val = train[(int(len(train)) * .7):]
        val_class = train_class[(int(len(train_label)) * .7):]

        train = train[:(int(len(train)) * .7)]
        train_class = train_class[:(int(len(train_label)) * .7)]

        '''
        Print some facts about my dataset
        '''
        train_hit_count = 0
        val_hit_count = 0

        for x in self.train_class:
            if x == classification1:
                train_hit_count+=1

        for z in self.val_class:
            if z == classification1:
                val_hit_count+=1


        print ("\n\n******************* Creating a Data Generator ************************\n")
        print ("\t\t  Number of train hits : ", train_hit_count)
        print ("\t\t Number of train misses : ", len(self.train_class) - train_hit_count)

        print ("\t\t   Number of val hits : ", val_hit_count)
        print ("\t\t  Number of val misses : ", len(self.val_class) - val_hit_count)
        print ("\n***********************************************************************\n\n")

        return self.train, self.train_class, self.val, self.val_class
