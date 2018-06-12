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
from matplotlib import pyplot as plt

config = configparser.ConfigParser()
config.read('../config.ini')

base_dir = config['DEFAULT']['BASE_DIR']

def start():
    vrep.simxFinish(-1) # just in case, close all opened connections
    clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) #start my Connection
    error_code =vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    return clientID, error_code

def collectImageData(ca_model, pn_model, clientID, states, input_type, use_ca):
    if use_ca:
        ca_model.eval()
    pn_model.train()

    list_of_images = []

    if (input_type == 0 or input_type ==2) and use_ca:
        vid_states = states[0]
        st_states = states[1]

    elif input_type == 1:
        if use_ca:
            vid_states = states[2]
            st_states = states[3]
        pn_vidstates = states[0]
        pn_ststates = states[1]
    elif input_type == 3:
        if use_ca:
            vid_states = states[0]
            st_states = states[1]
            pn_states = states[2]
        else:
            pn_states = states[0]

    collector = []
    collector2 = []
    if clientID!=-1:
        err, tracer_handle = vrep.simxGetObjectHandle(clientID, 'LineTracer', vrep.simx_opmode_oneshot_wait)
        res,v0=vrep.simxGetObjectHandle(clientID,'Vision_sensor',vrep.simx_opmode_oneshot_wait)
        res,v1=vrep.simxGetObjectHandle(clientID,'PassiveVision_sensor',vrep.simx_opmode_oneshot_wait)
        ret_code, left_handle = vrep.simxGetObjectHandle(clientID,'DynamicLeftJoint', vrep.simx_opmode_oneshot_wait)
        ret_code, right_handle = vrep.simxGetObjectHandle(clientID,'DynamicRightJoint', vrep.simx_opmode_oneshot_wait)
        ret_code, base_handle = vrep.simxGetObjectHandle(clientID, 'LineTracerBase', vrep.simx_opmode_oneshot_wait)

        res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_streaming)
        ret_code, euler_angles = vrep.simxGetObjectOrientation(clientID, tracer_handle, -1, vrep.simx_opmode_streaming)
        t_end = time.time() + 2.8
        count = 0
        action = 0
        inference_counter = 0

        while (vrep.simxGetConnectionId(clientID)!=-1 and time.time() < t_end):
            res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_buffer)

            if res==vrep.simx_return_ok:

                img = np.array(image,dtype=np.uint8)
                img.resize([resolution[1],resolution[0],3])
                rotate_img = img.copy()
                rotate_img = np.flipud(img)
                list_of_images.append(rotate_img)

                ret_code, pos = vrep.simxGetObjectPosition(clientID, tracer_handle, -1, vrep.simx_opmode_oneshot)
                ret_code, velo, angle_velo = vrep.simxGetObjectVelocity(clientID, tracer_handle, vrep.simx_opmode_oneshot)
                ret_code, euler_angles = vrep.simxGetObjectOrientation(clientID, tracer_handle, -1, vrep.simx_opmode_buffer)
                collector.append([pos[0], pos[1], pos[2], velo[0], velo[1], velo[2], angle_velo[0], angle_velo[1], angle_velo[2], euler_angles[0], euler_angles[1], euler_angles[2], action])
                collector2.append([pos[0], pos[1], pos[2], velo[0], velo[1], velo[2], euler_angles[0], euler_angles[1], euler_angles[2], action])
                if use_ca:
                    torch_vid = torch.from_numpy(np.transpose(np.expand_dims((list_of_images[-1]).astype('float'),axis=0), (0, 3, 1, 2)))
                    torch_st = torch.from_numpy(np.asarray(collector2[-1]).astype('float'))
                    vid_to_ca = Variable(torch_vid.float().cuda())
                    st_to_ca = Variable(torch_st.float().cuda())
                    output, vid_states, st_states = ca_model(vid_to_ca, st_to_ca, vid_states, st_states)
                    output.detach_()

                inference_counter +=1
                if input_type == 0:
                    if count == 0:
                        a = torch.from_numpy(list_of_images[-1].flatten()).float().cuda()
                        b = torch.from_numpy(list_of_images[-1].flatten()).float().cuda()
                        c =torch.from_numpy(np.asarray(collector[-1]).astype('float')).float().cuda()
                        if use_ca:
                            d = torch.squeeze(output.data).float().cuda()
                            input_to_model = torch.cat([a, b, c, d])
                        else:
                            input_to_model = torch.cat([a, b, c])
                    else:
                        a = torch.from_numpy(list_of_images[-1].flatten()).float().cuda()
                        b = torch.from_numpy(list_of_images[-2].flatten()).float().cuda()
                        c =torch.from_numpy(np.asarray(collector[-1]).astype('float')).float().cuda()
                        if use_ca:
                            d = torch.squeeze(output.data).float().cuda()
                            input_to_model = torch.cat([a, b, c, d])
                        else:
                            input_to_model = torch.cat([a, b, c])
                    input_to_model = Variable(input_to_model)
                    out = pn_model(input_to_model)
                elif input_type == 1:
                    pn_vid = torch.from_numpy(np.transpose(np.expand_dims((list_of_images[-1]).astype('float'),axis=0), (0, 3, 1, 2)))
                    vid_input = Variable(pn_vid.float().cuda())
                    c = torch.from_numpy(np.asarray(collector[-1]).astype('float')).float().cuda()
                    if use_ca:
                        d = torch.squeeze(output.data).float().cuda()
                        st_input = Variable(torch.cat([c, d]).unsqueeze(0))
                    else:
                        st_input = Variable(c.unsqueeze(0))
                    out, pn_vidstates, pn_ststates = pn_model(vid_input, st_input, pn_vidstates, pn_ststates)
                elif input_type == 2:
                    #stack two images together and I need to verify the images being stacked are reasonable
                    if count == 0:
                        stacked_img = np.concatenate(
                            (np.transpose(np.expand_dims((list_of_images[-1]).astype('float'),axis=0), (0, 3, 1, 2)),
                            np.transpose(np.expand_dims((list_of_images[-1]).astype('float'),axis=0), (0, 3, 1, 2))), axis=1)
                        vid_input = Variable(torch.from_numpy(stacked_img).float().cuda())
                        c = torch.from_numpy(np.asarray(collector[-1]).astype('float')).float().cuda()
                        if use_ca:
                            d = torch.squeeze(output.data).float().cuda()
                            st_input = Variable(torch.cat([c, d]).unsqueeze(0))
                        else:
                            st_input = Variable(c.unsqueeze(0))
                    else:
                        stacked_img = np.concatenate(
                            (np.transpose(np.expand_dims((list_of_images[-1]).astype('float'),axis=0), (0, 3, 1, 2)),
                            np.transpose(np.expand_dims((list_of_images[-2]).astype('float'),axis=0), (0, 3, 1, 2))), axis=1)
                        vid_input = Variable(torch.from_numpy(stacked_img).float().cuda())
                        c = torch.from_numpy(np.asarray(collector[-1]).astype('float')).float().cuda()
                        if use_ca:
                            d = torch.squeeze(output.data).float().cuda()
                            st_input = Variable(torch.cat([c, d]).unsqueeze(0))
                        else:
                            st_input = Variable(c.unsqueeze(0))
                    out = pn_model(st_input, vid_input)

                elif input_type == 3:
                    if count == 0:
                        a = torch.from_numpy(list_of_images[-1].flatten()).float().cuda()
                        b = torch.from_numpy(list_of_images[-1].flatten()).float().cuda()
                        c =torch.from_numpy(np.asarray(collector[-1]).astype('float')).float().cuda()

                        if use_ca:
                            d = torch.squeeze(output.data).float().cuda()
                            input_to_model = Variable(torch.cat([a, b, c, d]).unsqueeze(0))
                        else:
                            input_to_model = Variable(torch.cat([a, b, c]).unsqueeze(0))

                    else:
                        a = torch.from_numpy(list_of_images[-1].flatten()).float().cuda()
                        b = torch.from_numpy(list_of_images[-1].flatten()).float().cuda()
                        c =torch.from_numpy(np.asarray(collector[-1]).astype('float')).float().cuda()
                        if use_ca:
                            d = torch.squeeze(output.data).float().cuda()
                            input_to_model = Variable(torch.cat([a, b, c, d]).unsqueeze(0))
                        else:
                            input_to_model = Variable(torch.cat([a, b, c]).unsqueeze(0))
                    out, pn_states = pn_model(input_to_model, pn_states)

                else:
                    print("Error 12")
                    sys.exit()

                m = Categorical(out)
                action = m.sample()
                pn_model.current_log_probs.append(m.log_prob(action))
                velo = (action -2)  * 15
                return_val = vrep.simxSetJointTargetVelocity(clientID, left_handle, velo, vrep.simx_opmode_oneshot)
                return_val2 = vrep.simxSetJointTargetVelocity(clientID, right_handle, velo, vrep.simx_opmode_oneshot_wait)
                count+=1

        return list_of_images, collector
    else:
        sys.exit()

def end(clientID):
    error_code =vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(clientID)
    return error_code

def view_image(image, name):
    plt.imshow(image, cmap='gray')
    plt.title(name)
    plt.show()

def create_convlstm_states(shape, batch):
    if torch.cuda.is_available():
        c = Variable(torch.zeros(batch, shape[0], shape[1], shape[2])).float().cuda()
        h = Variable(torch.zeros(batch, shape[0], shape[1], shape[2])).float().cuda()
    else:
        c = Variable(torch.zeros(batch, shape[0], shape[1], shape[2])).float()
        h = Variable(torch.zeros(batch, shape[0], shape[1], shape[2])).float()
    return (h, c)

def create_lstm_states(shape, batch):

    if torch.cuda.is_available():
        c = Variable(torch.zeros(batch, shape).float().cuda())
        h = Variable(torch.zeros(batch, shape).float().cuda())
    else:
        c = Variable(torch.zeros(batch, shape).float())
        h = Variable(torch.zeros(batch, shape).float())
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

def single_simulation_noca(pn_model, n_iter, txt_file_counter, inp_type):

    if inp_type == 0:
        input_pn = Variable(torch.from_numpy(np.zeros(64*64*3*2+13)).float().cuda())
        pn_model(input_pn)
        states = []

    elif inp_type == 1:
        vid_input_to_pn = Variable(torch.from_numpy(np.zeros((1, 3, 64, 64))).float().cuda())
        st_input_to_pn = Variable(torch.from_numpy(np.zeros((1, 13))).float().cuda())

        pn_vid_states, pn_st_states = create_recurrent_states(pn_model, 1)
        pn_model(vid_input_to_pn, st_input_to_pn, pn_vid_states, pn_st_states)

        states = [pn_vid_states, pn_st_states]

    elif inp_type == 2:
        vid_input_to_pn = Variable(torch.from_numpy(np.zeros((1, 6, 64, 64))).float().cuda())
        st_input_to_pn = Variable(torch.from_numpy(np.zeros((1, 13))).float().cuda())

        pn_model(st_input_to_pn, vid_input_to_pn)
        states = []

    elif inp_type == 3:
        prev_0 = create_lstm_states(pn_model.h_0_sz, 1)
        prev_1 = create_lstm_states(pn_model.h_1_sz, 1)
        prev_2 = create_lstm_states(pn_model.h_2_sz, 1)

        pn_prev_states = [prev_0, prev_1, prev_2]
        input_pn = Variable(torch.from_numpy(np.zeros(64*64*3*2+13)).float().cuda().unsqueeze(0))
        pn_model(input_pn, pn_prev_states)

        states = [pn_prev_states]
    else:
        print("need to implement new input type")
        sys.exit()

    clientID, start_error = start()
    image_array, state_array = collectImageData(None, pn_model, clientID, states, inp_type, False) #store these images
    end_error = end(clientID)
    state = np.asarray(state_array).astype(float)

    return state

def single_simulation(ca_model, pn_model, n_iter, txt_file_counter, inp_type):

    vid_input_to_model = Variable(torch.from_numpy(np.zeros((1, 3, 64, 64))).float().cuda())
    st_input_to_model = Variable(torch.from_numpy(np.zeros(10)).float().cuda())
    vid_states, st_states = create_recurrent_states(ca_model, 1)
    ca_model(vid_input_to_model, st_input_to_model, vid_states, st_states)

    if inp_type == 0:
        input_pn = Variable(torch.from_numpy(np.zeros(64*64*3*2+10+13)).float().cuda())

        pn_model(input_pn)
        states = [vid_states, st_states]
    elif inp_type == 1:
        vid_input_to_pn = Variable(torch.from_numpy(np.zeros((1, 3, 64, 64))).float().cuda())
        st_input_to_pn = Variable(torch.from_numpy(np.zeros((1, 23))).float().cuda())

        pn_vid_states, pn_st_states = create_recurrent_states(pn_model, 1)
        pn_model(vid_input_to_pn, st_input_to_pn, pn_vid_states, pn_st_states)

        states = [pn_vid_states, pn_st_states, vid_states, st_states]
    elif inp_type == 2:
        vid_input_to_pn = Variable(torch.from_numpy(np.zeros((1, 6, 64, 64))).float().cuda())
        st_input_to_pn = Variable(torch.from_numpy(np.zeros((1, 23))).float().cuda())

        pn_model(st_input_to_pn, vid_input_to_pn)

        states = [vid_states, st_states]
    elif inp_type == 3:
        prev_0 = create_lstm_states(pn_model.h_0_sz, 1)
        prev_1 = create_lstm_states(pn_model.h_1_sz, 1)
        prev_2 = create_lstm_states(pn_model.h_2_sz, 1)

        pn_prev_states = [prev_0, prev_1, prev_2]
        input_pn = Variable(torch.from_numpy(np.zeros(64*64*3*2+13+10)).float().cuda().unsqueeze(0))
        pn_model(input_pn, pn_prev_states)

        states = [vid_states, st_states, pn_prev_states]
    else:
        print("need to implement new input type")
        sys.exit()

    clientID, start_error = start()
    image_array, state_array = collectImageData(ca_model, pn_model, clientID, states, inp_type, True) #store these images
    end_error = end(clientID)
    state = np.asarray(state_array).astype(float)
    return state

def execute_exp(ca_model, pn_model, iter_start, iter_end, input_type, use_ca):
    txt_file_counter = 1
    lst = []
    for current_iteration in range(iter_start, iter_end):
        if use_ca:
            state = single_simulation(ca_model, pn_model, current_iteration, txt_file_counter, input_type)
        else:
            state = single_simulation_noca(pn_model, current_iteration, txt_file_counter, input_type)
        lst.append(state)
        txt_file_counter+=1
    return lst
