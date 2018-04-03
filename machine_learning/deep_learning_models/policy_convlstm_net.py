from conv_lstm_cell import StatefulConv2dLSTMCell
import torch
import torch.nn as nn
import torch.nn.functional as F

'''This network takes in the current state and predicts future collisions'''

class ConvLSTMPolicyNet(nn.Module):
    def __init__(self, input_shp_vid,
                        input_shp_st,
                        hidden_0,
                        hidden_1,
                        hidden_2,
                        hidden_out,
                        no_filters,
                        kernel_size,
                        strides,
                        output_shp,
                        padding=0,
                        dropout_rte=0):
        super(ConvLSTMPolicyNet, self).__init__()
        print("Initializing AnticipationNet")

        self.convlstm_0 = StatefulConv2dLSTMCell(input_shp_vid, no_filters[0], kernel_size, strides, pad=padding)
        self.convlstm_1 = StatefulConv2dLSTMCell(self.convlstm_0.output_shape, no_filters[1], kernel_size, strides, pad=padding)
        self.convlstm_2 = StatefulConv2dLSTMCell(self.convlstm_1.output_shape, no_filters[2], kernel_size, strides, pad=padding)

        self.LSTM_0 = nn.LSTMCell(input_shp_st, hidden_0)
        self.LSTM_1 = nn.LSTMCell(hidden_0, hidden_1)
        self.LSTM_2 = nn.LSTMCell(hidden_1, hidden_2)
        self.h_0_sz = hidden_0
        self.h_1_sz = hidden_1
        self.h_2_sz = hidden_2

        flat = self.convlstm_2.output_shape[0] * self.convlstm_2.output_shape[1] * self.convlstm_2.output_shape[2] + hidden_2

        self.dropout = nn.Dropout(dropout_rte)
        self.fcn1 = nn.Linear(flat, hidden_out)
        self.fcn2 = nn.Linear(hidden_out , output_shp)

        self.saved_log_probs = []
        self.updated_log_probs = []
        self.rewards = []
        self.reset_locations = []

    def forward(self, vid_x, st_x, vid_states, st_states):

        hx_0, cx_0 = self.convlstm_0(vid_x, [vid_states[0][0] ,vid_states[0][1]])
        hx_1, cx_1 = self.convlstm_1(hx_0, (vid_states[1][0] ,vid_states[1][1]))
        hx_2, cx_2 = self.convlstm_2(hx_1, (vid_states[2][0] ,vid_states[2][1]))

        h_0, c_0 = self.LSTM_0(st_x, (st_states[0][0], st_states[0][1]))
        h_1, c_1 = self.LSTM_1(h_0, (st_states[1][0], st_states[1][1]))
        h_2, c_2 = self.LSTM_2(h_1, (st_states[2][0], st_states[2][1]))

        hx_2_flat = hx_2.view(hx_2.size(0), -1)

        h_2 = h_2.view(h_2.size(0), -1)

        concat = torch.cat((hx_2_flat, h_2), dim=1)


        dropped = self.dropout(concat) #use dropout on flattened output of convlstm cell
        h_out = F.relu(self.fcn1(dropped))
        y = F.softmax(self.fcn2(h_out), dim=0) #regress the outputs

        return y, [[hx_0, cx_0], [hx_1, cx_1], [hx_2, cx_2]], [[h_0, c_0], [h_1, c_1], [h_2, c_2]]
