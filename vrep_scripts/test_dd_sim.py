import vrep
import sys
import time
import numpy as np
from scipy.misc import imsave
import random
import scipy.io as sio
import scipy
from scipy import ndimage
import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
config = configparser.ConfigParser()
config.read('../config.ini')

base_dir = config['DEFAULT']['BASE_DIR']

def start():
    vrep.simxFinish(-1) # just in case, close all opened connections
    clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) #start my Connection
    error_code = vrep.simxSynchronous(clientID, True)
    error_code =vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)
    return clientID, error_code

def collectImageData(clientID):

    list_of_images = []

    collector = []
    col_lst = []
    if clientID!=-1:
        res,v0=vrep.simxGetObjectHandle(clientID,'Vision_sensor',vrep.simx_opmode_oneshot_wait)
        res,v1=vrep.simxGetObjectHandle(clientID,'PassiveVision_sensor',vrep.simx_opmode_oneshot_wait)
        ret_code, left_handle = vrep.simxGetObjectHandle(clientID,'DynamicLeftJoint', vrep.simx_opmode_oneshot_wait)
        ret_code, right_handle = vrep.simxGetObjectHandle(clientID,'DynamicRightJoint', vrep.simx_opmode_oneshot_wait)
        ret_code, base_handle = vrep.simxGetObjectHandle(clientID, 'LineTracerBase', vrep.simx_opmode_oneshot_wait)

        res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_streaming)
        ret_code, euler_angles = vrep.simxGetObjectOrientation(clientID, base_handle, -1, vrep.simx_opmode_streaming)

        count = 0
        action = 2
        inference_counter = 0
        steps = 30
        delay=0

        while (vrep.simxGetConnectionId(clientID)!=-1 and count < steps):
            tim = time.time()

            '''The amount of time for inference for my model is .05 - .1 seconds'''
            _x = np.random.randint(0, 10)
            _x+=10
            for i in range(_x):
                vrep.simxSynchronousTrigger(clientID)
                col_lst.append(detectCollisionSignal(clientID))

            res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_buffer)
            if res==vrep.simx_return_ok:

                img = np.array(image,dtype=np.uint8)
                img.resize([resolution[1],resolution[0],3])
                rotate_img = img.copy()
                rotate_img = np.flipud(img)
                list_of_images.append(rotate_img)

                ret_code, pos = vrep.simxGetObjectPosition(clientID, base_handle, -1, vrep.simx_opmode_oneshot)
                ret_code, velo, angle_velo = vrep.simxGetObjectVelocity(clientID, base_handle, vrep.simx_opmode_oneshot)
                ret_code, euler_angles = vrep.simxGetObjectOrientation(clientID, base_handle, -1, vrep.simx_opmode_buffer)

                collector.append([pos[0], pos[1], pos[2], velo[0], velo[1], velo[2], angle_velo[0], angle_velo[1], angle_velo[2], euler_angles[0], euler_angles[1], euler_angles[2], action])

            if delay <= 0 :
                action = np.random.randint(0, 5)
                delay = np.random.randint(2,10)
                velo = (action -2)  * 15

                return_val = vrep.simxSetJointTargetVelocity(clientID, left_handle, velo, vrep.simx_opmode_oneshot)
                return_val2 = vrep.simxSetJointTargetVelocity(clientID, right_handle, velo, vrep.simx_opmode_oneshot_wait)
            delay-=1
            count+=1
        return list_of_images, collector, col_lst
    else:
        sys.exit()

def end(clientID):
    error_code =vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(clientID)
    return error_code

def detectCollisionSignal(clientID):
    collision_str = "collision_signal"
    detector = vrep.simxGetIntegerSignal(clientID, collision_str, vrep.simx_opmode_oneshot_wait)

    if detector[1] == 1:
        return 1
    else:
        return 0

def single_simulation():

    states = None
    clientID, start_error = start()
    image_array, state_array, col_lst = collectImageData(clientID) #store these images
    col_sig = detectCollisionSignal(clientID)
    end_error = end(clientID)
    state = np.asarray(state_array).astype(float)

    return state, col_sig, col_lst


def execute_exp():
    txt_file_counter = 1
    lst = []
    states, col_sig, col_lst = single_simulation()
    lst.append(states)

    return lst, col_lst
