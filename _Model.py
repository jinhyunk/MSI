import torch
import torch.nn as nn

from _Calc import *
from _utils import * 

class Encoder_champ(nn.Module):
    def __init__(self,
                 #Param
                 init_s_wr=40.0,
                 emb_size = 4,
                 out_size = 1,
                 # Value
                 c_wr=0.50,
                 ):
        super().__init__()
        self.c_wr = c_wr
        self.s_wr = nn.Parameter(torch.tensor(init_s_wr, dtype=torch.float32))
        self.normalize = normalize_winrate
        self.embedding = nn.Sequential(
            nn.Linear(2, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, out_size),
            nn.ReLU()  # 0~1 사이 값으로 정규화
        )

    def forward(self, wr, pb):
        wr_norm = self.normalize(wr,self.s_wr,self.c_wr)
        pb = torch.tensor(pb, dtype=torch.float32)
        enc_input = torch.stack([wr_norm,pb])
        output = self.embedding(enc_input)

        return output
    
class Encoder_player(nn.Module):
    def __init__(self,
                 # parameter
                 init_s_wr=10.0,
                 init_s_game = 0.2,

                 # value
                 c_wr=0.50,
                 emb_size = 8,
                 mim_game = 5,
                 out_size = 4,
                 ):
        super().__init__()
        self.s_wr = nn.Parameter(torch.tensor(init_s_wr))
        self.c_wr = c_wr
        
        self.s_game = nn.Parameter(torch.tensor(init_s_game))
        self.mim_game = mim_game
        
        self.normalize_wr = normalize_winrate
        self.normalize_ms = normalize_mastery

        self.embedding = nn.Sequential(
            nn.Linear(2, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, out_size),
            nn.ReLU()  # 0~1 사이 값으로 정규화
        )
    
    def forward(self,game,wr):
        wr_norm = self.normalize_wr(wr,self.s_wr,self.c_wr)
        mastery = self.normalize_ms(game,self.s_game,self.mim_game)
        enc_input = torch.stack([mastery,wr_norm])

        output = self.embedding(enc_input)

        return output
    
class Encoder_region(nn.Module):
    def __init__(self,
                 n_list=3,
                 emb_size = 4,
                 out_size = 2,
                 ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(n_list, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, out_size),
            nn.ReLU()  # 0~1 사이 값으로 정규화
        )

    def forward(self,input_stack):
        output = self.embedding(input_stack)

        return output

class Encoder_position(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.attn = nn.Linear(output_dim, 1)  # Attention score per position

    def forward(self, rank,lg,player):
        x = torch.cat([rank, lg, player], dim=1)
        h = self.mlp(x)                # (5, 8)
        attn_scores = self.attn(h)     # (5, 1)
        attn_weights = torch.softmax(attn_scores, dim=0)  # (5, 1)
        out = (attn_weights * h).sum(dim=0, keepdim=True)  # (1, 8)
        return out

class Encoder_ELO(nn.Module):
    def __init__(self,
                 emb_size = 16,
                 out_size = 8,
                 mu = 1500,
                 sigma = 200,
                 ):
        super().__init__()
        self.normalize = normalize_ELO
        self.mu = mu
        self.sigma = sigma
        self.embedding = nn.Sequential(
            nn.Linear(1, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, out_size),
            nn.ReLU()  # 0~1 사이 값으로 정규화
        )

    def forward(self,ELO):
        ELO_nm = self.normalize(ELO,self.mu,self.sigma)
        output = self.embedding(ELO_nm)

        return output
    
class MLP(nn.Module):
    def __init__(self,
                 in_size = 16,
                 emb_size = 16,
                 out_size = 1
                 ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, out_size),
            nn.ReLU()  # 0~1 사이 값으로 정규화
        )

    def forward(self,input):
        output = self.embedding(input)

        return output