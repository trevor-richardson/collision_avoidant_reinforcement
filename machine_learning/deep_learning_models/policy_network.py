import torch
import torch.nn as nn
import torch.nn.functional as F


'''Simple Recurrent Policy Network'''
class Policy_Network(nn.Module):
    def __init__(self, input_shp, hidden1_shp, hidden2_shp, hidden3_shp, output_shp, dropout_rte=0):
        super(Policy_Network, self).__init__()
        print("Initializing Recurrent Policy Network")

        self.lstm_1 = nn.LSTMCell(input_shp, hidden1_shp)
        self.lstm_2 = nn.LSTMCell(hidden1_shp, hidden2_shp)
        self.lstm_3 = nn.LSTMCell(hidden2_shp, hidden3_shp)
        self.dropout = nn.Dropout(dropout_rte)
        self.fcn1 = nn.Linear(hidden3_shp, output_shp) #outputshp represents the number of possible actions robot can take

    def forward(self, x, states):
        hx_0, cx_0 = self.lstm_1(x, states[0])
        hx_1, cx_1 = self.lstm_2(hx_0, states[1])
        hx_2, cx_2 = self.lstm_3(hx_1, states[2])
        dropped = self.dropout(hx_2)
        x = self.fcn1(dropped)
        return x, [[hx_0, cx_0], [hx_1, cx_1], [hx_2, cx_2]]
