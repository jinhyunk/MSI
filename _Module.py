import torch
import torch.nn as nn
import numpy as np

class Module_champ_tear(nn.Module):
    def __init__(self,
                 c_wr=0.50,
                 c_pb_1=0.20, c_pb_2=0.80,
                 init_s_wr=40.0,
                 init_s_pb_1=25.0, init_s_pb_2=5.0,
                 init_w_pb=0.55,
                 init_w_score=0.9):
        super().__init__()
        
        # 고정값
        self.c_wr = c_wr
        self.c_pb_1 = c_pb_1
        self.c_pb_2 = c_pb_2

        # 학습 파라미터
        self.s_wr = nn.Parameter(torch.tensor(init_s_wr, dtype=torch.float32))
        self.s_pb_1 = nn.Parameter(torch.tensor(init_s_pb_1, dtype=torch.float32))
        self.s_pb_2 = nn.Parameter(torch.tensor(init_s_pb_2, dtype=torch.float32))
        self.w_pb = nn.Parameter(torch.tensor(init_w_pb, dtype=torch.float32))
        self.w_score = nn.Parameter(torch.tensor(init_w_score, dtype=torch.float32))

    def normalize_winrate(self, wr):
        if wr != -1.0:
            wr = torch.tensor(wr, dtype=torch.float32)
        else:
            wr = torch.tensor(0.30, dtype=torch.float32)

        return 1 / (1 + torch.exp(-self.s_wr * (wr - self.c_wr)))

    def normalize_pickban(self, pb):
        pb = torch.tensor(pb, dtype=torch.float32)
        s1 = 1 / (1 + torch.exp(-self.s_pb_1 * (pb - self.c_pb_1)))
        s2 = 1 / (1 + torch.exp(-self.s_pb_2 * (pb - self.c_pb_2)))
        return self.w_pb * s1 + (1 - self.w_pb) * s2

    def forward(self, wr, pb):
        wr_norm = self.normalize_winrate(wr)
        pb_norm = self.normalize_pickban(pb)
        
        return self.w_score * wr_norm + (1 - self.w_score) * pb_norm

class Module_team(nn.Module):
    def __init__(self,
                 ):
        super().__init__()

        self.module_champ_rank = Module_champ_tear(init_s_wr=40.0,init_w_score=0.7)
        self.module_champ_lg = Module_champ_tear(init_s_wr=10.0,init_w_score=0.1)
    
    def forward(self,data_player,data_champ):
        champ_tear_rank = self.module_champ_rank(data_champ)
        champ_tear_lg = self.module_champ_lg(data_champ)