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

x_list_of_positions = np.random.normal(0, 1.0, 3000)
y_list_of_positions = np.random.normal(-12 , 1.0, 3000)
x_list_of_positions0 = np.random.normal(1, 1.0, 3000)
y_list_of_positions0 = np.random.normal(-12 , 1.0, 3000)
x_list_of_positions1 = np.random.normal(-1, 1.0, 3000)
y_list_of_positions1 = np.random.normal(-12 , 1.0, 3000)
x_list_of_positions2 = np.random.normal(2, 1.0, 3000)
y_list_of_positions2 = np.random.normal(-12 , 1.0, 3000)
x_list_of_positions3 = np.random.normal(-2, 1.0, 3000)
y_list_of_positions3 = np.random.normal(-12 , 1.0, 3000)
x_list_of_positions4 = np.random.normal(0, 1.0, 3000)
y_list_of_positions4 = np.random.normal(-12 , 1.0, 3000)
x_list_of_positions5 = np.random.normal(0, 1.0, 3000)
y_list_of_positions5 = np.random.normal(-12 , 1.0, 3000)
x_list_of_positions6 = np.random.normal(0, 1.0, 3000)
y_list_of_positions6 = np.random.normal(-12 , 1.0, 3000)
z_permanent = .2555

with open(base_dir + '/vrep_scripts/saved_vel_pos_data/current_position.txt', "w") as new_pos_file:
    print(x_list_of_positions[0], file=new_pos_file)
    print(y_list_of_positions[0], file=new_pos_file)
    print(z_permanent, file=new_pos_file)

with open(base_dir + '/vrep_scripts/saved_vel_pos_data/current_position0.txt', "w") as new_pos_file:
    print(x_list_of_positions0[0], file=new_pos_file)
    print(y_list_of_positions0[0], file=new_pos_file)
    print(z_permanent, file=new_pos_file)

with open(base_dir + '/vrep_scripts/saved_vel_pos_data/current_position1.txt', "w") as new_pos_file:
    print(x_list_of_positions1[0], file=new_pos_file)
    print(y_list_of_positions1[0], file=new_pos_file)
    print(z_permanent, file=new_pos_file)

with open(base_dir + '/vrep_scripts/saved_vel_pos_data/current_position2.txt', "w") as new_pos_file:
    print(x_list_of_positions2[0], file=new_pos_file)
    print(y_list_of_positions2[0], file=new_pos_file)
    print(z_permanent, file=new_pos_file)

with open(base_dir + '/vrep_scripts/saved_vel_pos_data/current_position3.txt', "w") as new_pos_file:
    print(x_list_of_positions3[0], file=new_pos_file)
    print(y_list_of_positions3[0], file=new_pos_file)
    print(z_permanent, file=new_pos_file)

with open(base_dir + '/vrep_scripts/saved_vel_pos_data/current_position4.txt', "w") as new_pos_file:
    print(x_list_of_positions4[0], file=new_pos_file)
    print(y_list_of_positions4[0], file=new_pos_file)
    print(z_permanent, file=new_pos_file)

with open(base_dir + '/vrep_scripts/saved_vel_pos_data/current_position5.txt', "w") as new_pos_file:
    print(x_list_of_positions5[0], file=new_pos_file)
    print(y_list_of_positions5[0], file=new_pos_file)
    print(z_permanent, file=new_pos_file)

with open(base_dir + '/vrep_scripts/saved_vel_pos_data/current_position6.txt', "w") as new_pos_file:
    print(x_list_of_positions6[0], file=new_pos_file)
    print(y_list_of_positions6[0], file=new_pos_file)
    print(z_permanent, file=new_pos_file)

def start():
    vrep.simxFinish(-1) # just in case, close all opened connections
    clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) #start my Connection
    error_code = vrep.simxSynchronous(clientID, True)
    error_code =vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)
    return clientID, error_code


def collectImageData(ca_model, pn_model, clientID, states, input_type, use_ca):

    list_of_images = []

    collector = []
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
        steps = 50
        delay=0

        while (vrep.simxGetConnectionId(clientID)!=-1 and count < steps):
            # time.sleep(1)
            tim = time.time()
            for i in range(10):
                vrep.simxSynchronousTrigger(clientID)

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
        return list_of_images, collector
    else:
        sys.exit()

def end(clientID):
    error_code =vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(clientID)
    return error_code

def detectCollisionSignal(clientID):
    detector = 0
    collision_str = "collision_signal"
    detector = vrep.simxGetIntegerSignal(clientID, collision_str, vrep.simx_opmode_oneshot_wait)
    start = time.time()
    while(time.time() < start +1):
        pass

    if detector[1] == 1:
        return 1
    else:
        return 0

def create_convlstm_states(shape, batch):
    if torch.cuda.is_available():
        c = Variable(torch.zeros(batch, shape[0], shape[1], shape[2]), volatile=True).float().cuda()
        h = Variable(torch.zeros(batch, shape[0], shape[1], shape[2]), volatile=True).float().cuda()
    else:
        c = Variable(torch.zeros(batch, shape[0], shape[1], shape[2]), volatile=True).float()
        h = Variable(torch.zeros(batch, shape[0], shape[1], shape[2]), volatile=True).float()
    return (h, c)

def create_lstm_states(shape, batch):
    if torch.cuda.is_available():
        c = Variable(torch.zeros(batch, shape).float().cuda(), volatile=True)
        h = Variable(torch.zeros(batch, shape).float().cuda(), volatile=True)
    else:
        c = Variable(torch.zeros(batch, shape).float(), volatile=True)
        h = Variable(torch.zeros(batch, shape).float(), volatile=True)
    return (h, c)

def create_recurrent_states(model, batch):
    prev0 = create_convlstm_states(model.convlstm_0.output_shape, batch)
    prev1 = create_convlstm_states(model.convlstm_1.output_shape, batch)
    prev2 = create_convlstm_states(model.convlstm_2.output_shape, batch)

    vid_states = [prev0, prev1, prev2]

    prev_0 = create_lstm_states(model.h_0_sz, batch)
    prev_1 = create_lstm_states(model.h_1_sz, batch)
    prev_2 = create_lstm_states(model.h_2_sz, batch)

    st_states = [prev_0, prev_1, prev_2]

    return vid_states, st_states

def single_simulation_noca(pn_model, n_iter, txt_file_counter, inp_type, get_collision):

    states = None

    clientID, start_error = start()
    image_array, state_array = collectImageData(None, pn_model, clientID, states, inp_type, False) #store these images
    if get_collision:
        col_sig = detectCollisionSignal(clientID)
    else:
        col_sig = None
    end_error = end(clientID)
    state = np.asarray(state_array).astype(float)

    return state, col_sig

def single_simulation(ca_model, pn_model, n_iter, txt_file_counter, inp_type, get_collision):

    states = None
    clientID, start_error = start()
    image_array, state_array = collectImageData(ca_model, pn_model, clientID, states, inp_type, True) #store these images
    if get_collision:
        col_sig = detectCollisionSignal(clientID)
    else:
        col_sig = None
    end_error = end(clientID)
    state = np.asarray(state_array).astype(float)

    return state, col_sig


def execute_exp(ca_model, pn_model, iter_start, iter_end, input_type, use_ca, get_collision):
    txt_file_counter = 1
    lst = []

    for current_iteration in range(iter_start, iter_end):
        if use_ca:
            states, col_sig = single_simulation(ca_model, pn_model, current_iteration, txt_file_counter, input_type, get_collision)
        else:
            states, col_sig = single_simulation_noca(pn_model, current_iteration, txt_file_counter, input_type, get_collision)
        lst.append(states)
    return lst, col_sig
