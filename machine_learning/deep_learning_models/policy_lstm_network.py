import torch
import torch.nn as nn
import torch.nn.functional as F

'''Simple Recurrent Policy Network'''
class Policy_LSTMNetwork(nn.Module):
    def __init__(self, input_shp,
                    num_neurons_0,
                    num_neurons_1,
                    num_neurons_2,
                    output_shp):
        super(Policy_LSTMNetwork, self).__init__()
        print("Initializing Policy Network")

        self.h_0_sz = num_neurons_0
        self.h_1_sz = num_neurons_1
        self.h_2_sz = num_neurons_2

        self.LSTM_0 = nn.LSTMCell(input_shp, num_neurons_0)
        self.LSTM_1 = nn.LSTMCell(num_neurons_0, num_neurons_1)
        self.LSTM_2 = nn.LSTMCell(num_neurons_1, num_neurons_2)
        self.output = nn.Linear(num_neurons_2, output_shp)

        self.saved_log_probs = []
        self.rewards = []
        self.reset_locations = []

    def forward(self, x, st_states):

        h_0, c_0 = self.LSTM_0(x, (st_states[0][0], st_states[0][1]))
        h_1, c_1 = self.LSTM_1(h_0, (st_states[1][0], st_states[1][1]))
        h_2, c_2 = self.LSTM_2(h_1, (st_states[2][0], st_states[2][1]))

        y = F.softmax(self.output(h_2), dim=1)

        return y, [[h_0, c_0], [h_1, c_1], [h_2, c_2]]
