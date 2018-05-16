from conv_lstm_cell import StatefulConv2dLSTMCell
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
This network takes in the current state and predicts future collisions
convlstm custom built with no lstm chain
'''

class ConvLSTMPolicyNet(nn.Module):
    def __init__(self, input_shp_vid,
                        input_shp_st,
                        no_filters,
                        kernel_size,
                        strides,
                        output_shp,
                        padding=0):
        super(ConvLSTMPolicyNet, self).__init__()
        print("Initializing AnticipationNet")

        self.convlstm_0 = StatefulConv2dLSTMCell(input_shp_vid, no_filters[0], kernel_size, strides, pad=padding)
        self.convlstm_1 = StatefulConv2dLSTMCell(self.convlstm_0.output_shape, no_filters[1], kernel_size, strides, pad=padding)
        self.convlstm_2 = StatefulConv2dLSTMCell(self.convlstm_1.output_shape, no_filters[2], kernel_size, strides, pad=padding)

        flat = self.convlstm_2.output_shape[0] * self.convlstm_2.output_shape[1] * self.convlstm_2.output_shape[2] + input_shp_st

        self.normalize_state_layer = nn.Linear(input_shp_st, input_shp_st)

        self.output_shp = nn.Linear(flat , output_shp)

        self.saved_log_probs = []
        self.rewards = []
        self.reset_locations = []
        self.current_log_probs = []

    def forward(self, vid_x, st_x, vid_states, st_states):

        hx_0, cx_0 = self.convlstm_0(vid_x, (vid_states[0][0] ,vid_states[0][1]))
        hx_1, cx_1 = self.convlstm_1(hx_0, (vid_states[1][0] ,vid_states[1][1]))
        hx_2, cx_2 = self.convlstm_2(hx_1, (vid_states[2][0] ,vid_states[2][1]))

        h = F.tanh(self.normalize_state_layer(st_x))

        flat = torch.cat((hx_2.view(hx_2.size(0), -1), h), dim=1)

        y = F.softmax(self.output(flat), dim=1)

        return y, [[hx_0, cx_0], [hx_1, cx_1], [hx_2, cx_2]], [[h_0, c_0], [h_1, c_1], [h_2, c_2]]
