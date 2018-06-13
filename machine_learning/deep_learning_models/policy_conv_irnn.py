from irnn import IRNNCell
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
'''Custom Conv Irnn Neural Network'''

class ConviRNNPolicy_Network(nn.Module):
    def __init__(self, input_shp_st,
                    inp_img_shp,
                    filter_0,
                    filter_1,
                    filter_2,
                    filter_size,
                    num_neurons_0,
                    num_neurons_1,
                    output_shp):
        super(ConviRNNPolicy_Network, self).__init__()
        print("Initializing Policy Network")

        #Analyzes image information
        self.img_inp = inp_img_shp

        self.conv0 = nn.Sequential(
            nn.Conv2d(inp_img_shp[0], filter_0, kernel_size=filter_size, stride=2, padding=2),
            nn.BatchNorm2d(filter_0),
            nn.ReLU())

        self.conv1 = nn.Sequential(
            nn.Conv2d(filter_0, filter_1, kernel_size=filter_size, stride=2, padding=2),
            nn.BatchNorm2d(filter_1),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(filter_1, filter_2, kernel_size=filter_size, stride=2, padding=2),
            nn.BatchNorm2d(filter_2),
            nn.ReLU())

        ''' conv0 padding of two and stride for now this is hard coded to work'''
        conv_flat_size = (inp_img_shp[1] // 4) * (inp_img_shp[2] // 4) * filter_2

        self.irnn_0 = IRNNCell(input_shp_st, num_neurons_0)
        self.irnn_cat = IRNNCell(1360, num_neurons_1)

        self.output = nn.Linear(num_neurons_1, output_shp)

        self.prev_irnn_0_h = None
        self.prev_irnn_cat_h = None
        self.hidden_size_0 = num_neurons_0
        self.hidden_size_1 = num_neurons_1

        self.saved_log_probs = []
        self.rewards = []
        self.reset_locations = []
        self.current_log_probs = []

    def reset(self, batch):
        if torch.cuda.is_available():
            self.prev_irnn_0_h = (Variable(torch.zeros(batch, self.hidden_size_0)).cuda())
            self.prev_irnn_cat_h = (Variable(torch.zeros(batch, self.hidden_size_1)).cuda())
        else:
            self.prev_irnn_0_h = (Variable(torch.zeros(batch, self.hidden_size_0)))
            self.prev_irnn_cat_h = (Variable(torch.zeros(batch, self.hidden_size_1)))


    def forward(self, x_st, x_img):
        himg_0 = self.conv0(x_img)
        himg_1 = self.conv1(himg_0)
        himg_2 = self.conv2(himg_1)

        self.prev_irnn_0_h = self.irnn_0(x_st, self.prev_irnn_0_h)

        flat = torch.cat((himg_2.view(himg_2.size(0), -1), self.prev_irnn_0_h), dim=1)

        self.prev_irnn_cat_h = self.irnn_cat(flat, self.prev_irnn_cat_h)

        y = F.softmax(self.output(self.prev_irnn_cat_h), dim=1)

        return y
