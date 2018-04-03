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
    error_code =vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    return clientID, error_code

def collectImageData(ca_model, pn_model, clientID, states, input_type):
    ca_model.eval()
    pn_model.train()

    list_of_images = []
    if input_type == 0:
        vid_states = states[0]
        st_states = states[1]

    elif input_type == 1:
        vid_states = states[2]
        st_states = states[3]
        pn_vidstates = states[0]
        pn_ststates = states[1]

    collector = []
    if clientID!=-1:
        res,v0=vrep.simxGetObjectHandle(clientID,'Vision_sensor',vrep.simx_opmode_oneshot_wait)
        res,v1=vrep.simxGetObjectHandle(clientID,'PassiveVision_sensor',vrep.simx_opmode_oneshot_wait)
        ret_code, left_handle = vrep.simxGetObjectHandle(clientID,'DynamicLeftJoint', vrep.simx_opmode_oneshot_wait)
        ret_code, right_handle = vrep.simxGetObjectHandle(clientID,'DynamicRightJoint', vrep.simx_opmode_oneshot_wait)
        ret_code, base_handle = vrep.simxGetObjectHandle(clientID, 'LineTracerBase', vrep.simx_opmode_oneshot_wait)

        res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_streaming)
        ret_code, euler_angles = vrep.simxGetObjectOrientation(clientID, base_handle, -1, vrep.simx_opmode_streaming)
        t_end = time.time() + 2.8
        count = 0
        action = 0

        while (vrep.simxGetConnectionId(clientID)!=-1 and time.time() < t_end):
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
                collector.append([pos[0], pos[1], pos[2], velo[0], velo[1], velo[2], euler_angles[0], euler_angles[1], euler_angles[2], action])

                torch_vid = torch.from_numpy(np.transpose(np.expand_dims((list_of_images[-1]).astype('float'),axis=0), (0, 3, 1, 2)))
                torch_st = torch.from_numpy(np.asarray(collector[-1]).astype('float'))
                vid_to_ca = Variable(torch_vid.float().cuda())
                st_to_ca = Variable(torch_st.float().cuda())
                output, vid_states, st_states = ca_model(vid_to_ca, st_to_ca, vid_states, st_states)
                output.detach_()

                if (count) % 15 == 0:
                    if input_type == 0:
                        if count == 0:
                            a = torch.from_numpy(list_of_images[-1].flatten()).float().cuda()
                            b = torch.from_numpy(list_of_images[-1].flatten()).float().cuda()
                            c =torch.from_numpy(np.asarray(collector[-1]).astype('float')).float().cuda()
                            d = torch.squeeze(output.data).float().cuda()
                            input_to_model = torch.cat([a, b, c, d])
                        else:
                            a = torch.from_numpy(list_of_images[-1].flatten()).float().cuda()
                            b = torch.from_numpy(list_of_images[-1].flatten()).float().cuda()
                            c =torch.from_numpy(np.asarray(collector[-1]).astype('float')).float().cuda()
                            d = torch.squeeze(output.data).float().cuda()
                            input_to_model = torch.cat([a, b, c, d])
                        input_to_model = Variable(input_to_model)
                        out = pn_model(input_to_model)
                    elif input_type == 1:
                        pn_vid = torch.from_numpy(np.transpose(np.expand_dims((list_of_images[-1]).astype('float'),axis=0), (0, 3, 1, 2)))
                        vid_input = Variable(pn_vid.float().cuda())
                        c = torch.from_numpy(np.asarray(collector[-1]).astype('float')).float().cuda()
                        d = torch.squeeze(output.data).float().cuda()
                        st_input = Variable(torch.cat([c, d]))
                        out, pn_vidstates, pn_ststates = pn_model(vid_input, st_input, pn_vidstates, pn_ststates)

                    elif input_type == 2:
                        print("Error 12")
                        sys.exit()

                    m = Categorical(out)
                    act = m.sample()
                    pn_model.saved_log_probs.append(m.log_prob(act))
                    action = (act -2)  * 15
                    return_val = vrep.simxSetJointTargetVelocity(clientID, left_handle, action, vrep.simx_opmode_oneshot)
                    return_val2 = vrep.simxSetJointTargetVelocity(clientID, right_handle, action, vrep.simx_opmode_oneshot_wait)

                count+=1

        return list_of_images, collector
    else:
        sys.exit()

def end(clientID):
    error_code =vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(clientID)
    return error_code

def write_to_hit_miss_txt(n_iter, txt_file_counter):
    filename_newpos = base_dir + '/vrep_scripts/saved_vel_pos_data/current_position.txt'
    filename_newpos0 = base_dir + '/vrep_scripts/saved_vel_pos_data/current_position0.txt'
    filename_newpos1 = base_dir + '/vrep_scripts/saved_vel_pos_data/current_position1.txt'
    filename_newpos2 = base_dir + '/vrep_scripts/saved_vel_pos_data/current_position2.txt'
    filename_newpos3 = base_dir + '/vrep_scripts/saved_vel_pos_data/current_position3.txt'
    filename_newpos4 = base_dir + '/vrep_scripts/saved_vel_pos_data/current_position4.txt'
    filename_newpos5 = base_dir + '/vrep_scripts/saved_vel_pos_data/current_position5.txt'
    filename_newpos6 = base_dir + '/vrep_scripts/saved_vel_pos_data/current_position6.txt'
    filename_miss = base_dir + '/vrep_scripts/saved_vel_pos_data/train/miss/miss' + str(txt_file_counter)
    filename_hit = base_dir + '/vrep_scripts/saved_vel_pos_data/train/hit/hit' + str(txt_file_counter)
    filename_get_velocity = base_dir + '/vrep_scripts/saved_vel_pos_data/velocity.txt'

    with open(filename_newpos, "w") as new_pos_file:
        print(x_list_of_positions[txt_file_counter + 1], file=new_pos_file)
        print(y_list_of_positions[txt_file_counter + 1], file=new_pos_file)
        print(z_permanent, file=new_pos_file)
    with open(filename_newpos0, "w") as new_pos_file:
        print(x_list_of_positions0[txt_file_counter + 1], file=new_pos_file)
        print(y_list_of_positions0[txt_file_counter + 1], file=new_pos_file)
        print(z_permanent, file=new_pos_file)
    with open(filename_newpos1, "w") as new_pos_file:
        print(x_list_of_positions1[txt_file_counter + 1], file=new_pos_file)
        print(y_list_of_positions1[txt_file_counter + 1], file=new_pos_file)
        print(z_permanent, file=new_pos_file)
    with open(filename_newpos2, "w") as new_pos_file:
        print(x_list_of_positions2[txt_file_counter + 1], file=new_pos_file)
        print(y_list_of_positions2[txt_file_counter + 1], file=new_pos_file)
        print(z_permanent, file=new_pos_file)
    with open(filename_newpos3, "w") as new_pos_file:
        print(x_list_of_positions3[txt_file_counter + 1], file=new_pos_file)
        print(y_list_of_positions3[txt_file_counter + 1], file=new_pos_file)
        print(z_permanent, file=new_pos_file)
    with open(filename_newpos4, "w") as new_pos_file:
        print(x_list_of_positions4[txt_file_counter + 1], file=new_pos_file)
        print(y_list_of_positions4[txt_file_counter + 1], file=new_pos_file)
        print(z_permanent, file=new_pos_file)
    with open(filename_newpos5, "w") as new_pos_file:
        print(x_list_of_positions5[txt_file_counter + 1], file=new_pos_file)
        print(y_list_of_positions5[txt_file_counter + 1], file=new_pos_file)
        print(z_permanent, file=new_pos_file)
    with open(filename_newpos6, "w") as new_pos_file:
        print(x_list_of_positions6[txt_file_counter + 1], file=new_pos_file)
        print(y_list_of_positions6[txt_file_counter + 1], file=new_pos_file)
        print(z_permanent, file=new_pos_file)

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

def single_simulation(ca_model, pn_model, n_iter, txt_file_counter, inp_type):

    vid_input_to_model = Variable(torch.from_numpy(np.zeros((1, 3, 64, 64))).float().cuda())
    st_input_to_model = Variable(torch.from_numpy(np.zeros(10)).float().cuda())
    vid_states, st_states = create_recurrent_states(ca_model, 1)
    ca_model(vid_input_to_model, st_input_to_model, vid_states, st_states)

    if inp_type == 0:
        input_pn = Variable(torch.from_numpy(np.zeros(64*64*3*2+10+10)).float().cuda())
        pn_model(input_pn)
        states = [vid_states, st_states]
    elif inp_type == 1:
        vid_input_to_pn = Variable(torch.from_numpy(np.zeros((1, 3, 64, 64))).float().cuda())
        st_input_to_pn = Variable(torch.from_numpy(np.zeros(20)).float().cuda())
        pn_vid_states, pn_st_states = create_recurrent_states(pn_model, 1)
        pn_model(vid_input_to_pn, st_input_to_pn, pn_vid_states, pn_st_states)
        states = [pn_vid_states, pn_st_states, vid_states, st_states]
    else:
        print("need to implement new input type")
        sys.exit()

    clientID, start_error = start()
    image_array, state_array = collectImageData(ca_model, pn_model, clientID, states, inp_type) #store these images
    end_error = end(clientID)
    state = np.asarray(state_array).astype(float)
    write_to_hit_miss_txt(n_iter, txt_file_counter)

    return state

def execute_exp(ca_model, pn_model, iter_start, iter_end, input_type):
    txt_file_counter = 1
    lst = []
    for current_iteration in range(iter_start, iter_end):
        state = single_simulation(ca_model, pn_model, current_iteration, txt_file_counter, input_type)
        lst.append(state)
        txt_file_counter+=1
    return lst
