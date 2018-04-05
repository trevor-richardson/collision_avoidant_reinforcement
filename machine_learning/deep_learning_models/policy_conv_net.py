import torch
import torch.nn as nn
import torch.nn.functional as F

'''Simple Policy Network using convolutional layers to analyze image input'''

class ConvPolicy_Network(nn.Module):
    def __init__(self, input_shp_st,
                    inp_img_shp,
                    filter_0,
                    filter_1,
                    filter_2,
                    filter_size,
                    num_neurons_0,
                    num_neurons_1,
                    num_neurons_2,
                    output_shp):
        super(ConvPolicy_Network, self).__init__()
        print("Initializing Policy Network")

        #Analyzes image information
        self.conv0 = nn.Conv2d(inp_img_shp[0], filter_0 ,filter_size, stride=2, padding=2)
        self.conv1 = nn.Conv2d(filter_0, filter_1, filter_size, stride=2, padding=2)
        self.conv2 = nn.Conv2d(filter_1, filter_2, filter_size, stride=2, padding=2)
        ''' conv0 padding of two and stride '''
        self.conv_flat_size = (inp_img_shp[1] / 4) * (inp_img_shp[2] / 4) * filter_2
        #analyzes state information
        self.lin_0 = nn.Linear(input_shp, num_neurons_0)
        self.lin_1 = nn.Linear(num_neurons_0, num_neurons_1)
        self.lin_2 = nn.Linear(num_neurons_1, num_neurons_2)

        #flattened dimension and num_neurons_2
        self.output = nn.Linear(num_neurons_2 + self.conv_flat_size, output_shp)

        self.saved_log_probs = []
        self.rewards = []
        self.reset_locations = []

    def forward(self, x_st, x_img):

        himg_0 = conv0(x_img)
        himg_1 = conv1(himg_0)
        himg_2 = conv2(himg_1)

        hst_0 = F.tanh(self.lin_0(x_st))
        hst_1 = F.tanh(self.lin_1(hst_0))
        hst_2 = F.tanh(self.lin_2(hst_1))

        flat = torch.cat((himg_2.view(himg_2.size(0), -1), hst_2), dim=1)
        y = F.softmax(self.output(flat), dim=0)

        return y
