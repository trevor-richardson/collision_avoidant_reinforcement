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

def collectImageData(model, clientID):
    #get my vision sensor working and print the data in the while loop below for 5 seconds after simulation begins
    list_of_images = []
    collector = []
    if clientID!=-1:
        res,v0=vrep.simxGetObjectHandle(clientID,'Vision_sensor',vrep.simx_opmode_oneshot_wait)
        res,v1=vrep.simxGetObjectHandle(clientID,'PassiveVision_sensor',vrep.simx_opmode_oneshot_wait)
        ret_code, left_handle = vrep.simxGetObjectHandle(clientID,'DynamicLeftJoint', vrep.simx_opmode_oneshot_wait)
        ret_code, right_handle = vrep.simxGetObjectHandle(clientID,'DynamicRightJoint', vrep.simx_opmode_oneshot_wait)
        ret_code, base_handle = vrep.simxGetObjectHandle(clientID, 'LineTracerBase', vrep.simx_opmode_oneshot_wait)

        res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_streaming)
        t_end = time.time() + 2.8
        collision_bool = False
        count = 0
        action = 0
        while (vrep.simxGetConnectionId(clientID)!=-1 and time.time() < t_end):
            res,resolution,image=vrep.simxGetVisionSensorImage(clientID,v0,0,vrep.simx_opmode_buffer)
            '''Execute policy network'''

            #convert the image add to numpy array we collect about 35 images per simulation
            if res==vrep.simx_return_ok:

                #got state data
                img = np.array(image,dtype=np.uint8)
                img.resize([resolution[1],resolution[0],3])
                rotate_img = img.copy()
                rotate_img = np.flipud(img)
                list_of_images.append(rotate_img)

                ret_code, pos = vrep.simxGetObjectPosition(clientID, base_handle, -1, vrep.simx_opmode_oneshot)
                ret_code, velo, angle_velo = vrep.simxGetObjectVelocity(clientID, base_handle, vrep.simx_opmode_oneshot)
                collector.append([pos[0], pos[1], pos[2], velo[0], velo[1], velo[2], action])

                if (count) % 75 == 0:
                    # print(np.concatenate((list_of_images[-1].flatten(), np.asarray(collector[-1]))).shape)
                    arr = np.concatenate((list_of_images[-1].flatten(), np.asarray(collector[-1]))).astype('float')
                    torch_arr = torch.from_numpy(arr)
                    input_to_model = Variable(torch_arr.float().cuda())
                    out = model(input_to_model)
                    m = Categorical(out)
                    act = m.sample()
                    model.saved_log_probs.append(m.log_prob(act))
                    action = (act -2)  * 15
                    return_val = vrep.simxSetJointTargetVelocity(clientID, left_handle, action, vrep.simx_opmode_oneshot)
                    return_val2 = vrep.simxSetJointTargetVelocity(clientID, right_handle, action, vrep.simx_opmode_oneshot_wait)
                else:
                    model.saved_log_probs.append(m.log_prob(act))

                count+=1
        return list_of_images, collector
    else:
        sys.exit()

def end(clientID):
    error_code =vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(clientID)
    return error_code

#check if there was a collision by calling globale variable set inside VREP
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

def writeImagesStatesToFiles(model, image_array, state_array, n_iter, collision_signal):
    reduced_image = []
    reduced_state = []

    time_dilation = round(np.random.normal(12, 3)) # make sure this system can work independent of the time dilation or hz of images coming in
    if time_dilation < 5:
        time_dilation == 5
    elif time_dilation > 20:
        time_dilation == 20 #set limits on how dilated the video it sees can be
    time_dilation = 10 #Leave this if you dont want to grab the right images

    reduced_image.append(image_array[0])
    reduced_state.append(state_array[0])
    model.updated_log_probs.append(model.saved_log_probs[0])
    print(len(image_array), len(state_array))

    for enumerator in range(len(image_array)):
        if enumerator % time_dilation == 0 and enumerator != 0:
            noise = random.uniform(0, 1) #dont grab every video at that exact offset
            if noise < .1:
                reduced_state.append(state_array[enumerator - 2])
                reduced_image.append(image_array[enumerator - 2])
                model.updated_log_probs.append(model.saved_log_probs[enumerator - 2])
            elif noise < .2:
                reduced_image.append(image_array[enumerator - 1])
                reduced_state.append(state_array[enumerator - 1])
                model.updated_log_probs.append(model.saved_log_probs[enumerator - 1])
            elif noise < .3:
                if enumerator + 1 < len(image_array):
                    reduced_state.append(state_array[enumerator + 1])
                    reduced_image.append(image_array[enumerator + 1])
                    model.updated_log_probs.append(model.saved_log_probs[enumerator + 1])
            elif noise < .4:
                if enumerator + 2 < len(image_array):
                    reduced_state.append(state_array[enumerator + 2])
                    reduced_image.append(image_array[enumerator + 2])
                    model.updated_log_probs.append(model.saved_log_probs[enumerator + 2])
            else:
                reduced_image.append(image_array[enumerator])
                reduced_state.append(state_array[enumerator])
                model.updated_log_probs.append(model.saved_log_probs[enumerator])

    # print("Cluster ", time_dilation, "  size of reduced array img and state ", len(reduced_image), len(reduced_state))
    selected_images = reduced_image[:70]
    selected_states = reduced_state[:70]
    model.updated_log_probs = model.updated_log_probs[:69]
    video_arr = np.concatenate([arr[np.newaxis] for arr in selected_images])
    video = np.moveaxis(video_arr, -1, 1).astype(float)
    state = np.asarray(selected_states).astype(float) #this is ready to be saved!


    test_or_train = random.uniform(0, 1)
    str_name_image = base_dir + '/data_generated/current_batch/image/' + str(n_iter) + 'collision'
    str_name_state = base_dir + '/data_generated/current_batch/state/' + str(n_iter) + 'collision'
    np.save(str_name_state, state)
    np.save(str_name_image, video)
    print(state.shape)
    print(video.shape)
    return video, state

def write_to_hit_miss_txt(n_iter, collision_signal, txt_file_counter):
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
    print("\n")

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


def single_simulation(model, n_iter, txt_file_counter):
    # print("####################################################################################################################")
    input_to_model = Variable(torch.from_numpy(np.zeros(64*64*3+7)).float().cuda(), volatile=True)
    out = model(input_to_model)

    clientID, start_error = start()
    image_array, state_array = collectImageData(model, clientID) #store these images
    collision_signal = detectCollisionSignal(clientID) #This records whether hit or miss
    end_error = end(clientID)
    if collision_signal:
        print("HIT")
    else:
        print("MISS")
    write_to_hit_miss_txt(n_iter, collision_signal, txt_file_counter)
    video, state = writeImagesStatesToFiles(model, image_array, state_array, n_iter, collision_signal)
    print("\n")
    return video, state

def execute_exp(model, iter_start, iter_end):
    txt_file_counter = 1
    lst = []
    for current_iteration in range(iter_start, iter_end):
        video, state = single_simulation(model, current_iteration, txt_file_counter)
        lst.append([video, state])
        txt_file_counter+=1
    return lst
