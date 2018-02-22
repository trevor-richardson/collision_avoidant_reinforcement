import torch
import torch.nn as nn
import torch.nn.functional as F

'''Define the neural network'''

class Deep_Dynamics(nn.Module):
    def __init__(self, input_shp, num_neurons_0, num_neurons_1, num_neurons_2, num_neurons_3, num_neurons_4, output_shp, dropout_rte=0):
        super(Deep_Dynamics, self).__init__()
        print("Initializing Deep Dynamics Model\n\n")

        self.h_0 = nn.Linear(input_shp, num_neurons_0)

        self.h_1 = nn.Linear(num_neurons_0, num_neurons_1)

        self.h_2 = nn.Linear(num_neurons_1, num_neurons_2)

        self.h_3 = nn.Linear(num_neurons_2, num_neurons_3)

        self.output = nn.Linear(num_neurons_3, output_shp)

        self.dropout = nn.Dropout(dropout_rte)

    def forward(self, x):

        drop_0 = self.dropout(F.tanh(self.h_0(x)))

        drop_1 = self.dropout(F.tanh(self.h_1(drop_0)))

        drop_2 = self.dropout(F.tanh(self.h_2(drop_1)))

        drop_3 = self.dropout(F.tanh(self.h_3(drop_2)))


        y = self.output(drop_3)

        return y
