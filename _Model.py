import torch
import torch.nn as nn

from _Calc import calc_mastery
from _utils import * 

class Encoder_champ(nn.Module):
    def __init__(self,
                 #Param
                 init_s_wr=40.0,
                 init_s_pb_1=25.0, init_s_pb_2=5.0,
                 emb_size = 4,
                 # Value
                 c_wr=0.50,
                 c_pb_1=0.20, c_pb_2=0.80,
                 w_pb=0.55,
                 ):
        super().__init__()
        self.c_wr = c_wr
        self.c_pb_1 = c_pb_1
        self.c_pb_2 = c_pb_2
        self.w_pb = w_pb

        self.s_wr = nn.Parameter(torch.tensor(init_s_wr, dtype=torch.float32))
        self.s_pb_1 = nn.Parameter(torch.tensor(init_s_pb_1, dtype=torch.float32))
        self.s_pb_2 = nn.Parameter(torch.tensor(init_s_pb_2, dtype=torch.float32))
        
        self.embedding = nn.Sequential(
            nn.Linear(2, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1),
            nn.ReLU()  # 0~1 사이 값으로 정규화
        )

    def normalize_winrate(self, wr):
        wr = torch.tensor(wr, dtype=torch.float32)
        return 1 / (1 + torch.exp(-self.s_wr * (wr - self.c_wr)))

    def normalize_pickban(self, pb):
        pb = torch.tensor(pb, dtype=torch.float32)
        s1 = 1 / (1 + torch.exp(-self.s_pb_1 * (pb - self.c_pb_1)))
        s2 = 1 / (1 + torch.exp(-self.s_pb_2 * (pb - self.c_pb_2)))
        return self.w_pb * s1 + (1 - self.w_pb) * s2

    def forward(self, wr, pb):
        wr_norm = self.normalize_winrate(wr)
        pb_norm = self.normalize_pickban(pb)
        enc_input = torch.stack([wr_norm,pb_norm])
        output = self.embedding(enc_input)

        return output
    
class Encoder_player(nn.Module):
    def __init__(self,
                 # parameter
                 init_s_wr=10.0,
                 init_s_game = 0.2,

                 # value
                 c_wr=0.50,
                 emb_size = 4,
                 mim_game = 5,
                 ):
        super().__init__()
        self.s_wr = nn.Parameter(torch.tensor(init_s_wr))
        self.c_wr = c_wr
        self.mim_game = mim_game
        self.s_game = nn.Parameter(torch.tensor(init_s_game))

        self.embedding = nn.Sequential(
            nn.Linear(2, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1),
            nn.ReLU()  # 0~1 사이 값으로 정규화
        )
    
    def normalize_winrate(self, wr):
        wr = torch.tensor(wr, dtype=torch.float32)
        return 1 / (1 + torch.exp(-self.s_wr * (wr - self.c_wr)))
    
    def normalize_mastery(self, game):
        return 1 / (1 + torch.exp(-self.s_game * (game - self.mim_game)))  # 0 ~ 1

    def forward(self,game,wr):
        wr_norm = self.normalize_winrate(wr)
        mastery = self.normalize_mastery(game)
        enc_input = torch.stack([mastery,wr_norm])

        output = self.embedding(enc_input)

        return output