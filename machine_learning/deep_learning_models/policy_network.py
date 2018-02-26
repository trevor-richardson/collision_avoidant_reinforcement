import torch
import torch.nn as nn
import torch.nn.functional as F


'''Simple Recurrent Policy Network'''
class Policy_Network(nn.Module):
    def __init__(self, input_shp, num_neurons_0, num_neurons_1, num_neurons_2, num_neurons_3, output_shp):
        super(Policy_Network, self).__init__()
        print("Initializing Policy Network")

        self.h_0 = nn.Linear(input_shp, num_neurons_0)
        self.h_1 = nn.Linear(num_neurons_0, num_neurons_1)
        self.h_2 = nn.Linear(num_neurons_1, num_neurons_2)
        self.h_3 = nn.Linear(num_neurons_2, num_neurons_3)
        self.output = nn.Linear(num_neurons_3, output_shp)

    def forward(self, x):

        drop_0 = F.tanh(self.h_0(x))
        drop_1 = F.tanh(self.h_1(drop_0))
        drop_2 = F.tanh(self.h_2(drop_1))
        drop_3 = F.tanh(self.h_3(drop_2))

        y = self.output(drop_3)

        return y
