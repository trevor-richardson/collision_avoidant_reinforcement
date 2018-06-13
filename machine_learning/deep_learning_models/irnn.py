import torch
import torch.nn as nn
import torch.nn.functional as F

class IRNNCell(nn.Module):
    def __init__(self, input_size,
                    hidden_size,
                    recurrent_act='relu',
                    weight_init=None,
                    reccurent_weight_init=None,
                    drop=None,
                    rec_drop=None):
        super(IRNNCell, self).__init__()

        print("Initializing IRNNCell")
        self.hidden_size = hidden_size
        if(weight_init==None):
            self.W_x = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_x = nn.init.xavier_normal_(self.W_x)
        else:
            self.W_x = nn.Parameter(torch.zeros(input_size, hidden_size))
            self.W_x = weight_init(self.W_x)

        self.U_h = torch.nn.Parameter(torch.eye(hidden_size))

        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.recurrent_act = recurrent_act

        if(drop==None):
            self.keep_prob = False
        else:
            self.keep_prob = True
            self.dropout = nn.Dropout(drop)
        if(rec_drop == None):
            self.rec_keep_prob = False
        else:
            self.rec_keep_prob = True
            self.rec_dropout = nn.Dropout(rec_drop)


        self.hidden_state = None


    def forward(self, X_t):
        h_t_previous = self.hidden_state

        if self.keep_prob:
            X_t = self.dropout(X_t)
        if self.rec_keep_prob:
            h_t_previous = self.rec_dropout(h_t_previous)
            c_t_previous = self.rec_dropout(c_t_previous)

        out = F.relu(
            torch.mm(X_t, self.W_x) + torch.mm(h_t_previous, self.U_h) + self.b
        )

        self.hidden_state = out
        return out