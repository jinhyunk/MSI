import torch
import torch.nn as nn

from _Calc import *
from _utils import * 


class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    
    def forward(self, input):
        return torch.sin(self.w0 * input)
    
def siren_init(m, w0=30.0, is_first=False):
    if isinstance(m, nn.Linear):
        with torch.no_grad():
            if is_first:
                # 첫 레이어
                num_input = m.weight.size(-1)
                m.weight.uniform_(-1 / num_input, 1 / num_input)
            else:
                # 이후 레이어
                num_input = m.weight.size(-1)
                bound = math.sqrt(6 / num_input) / w0
                m.weight.uniform_(-bound, bound)
            if m.bias is not None:
                m.bias.fill_(0)

class Encoder_champ(nn.Module):
    def __init__(self,
                 init_s_wr=40.0,
                 emb_size = 4,
                 out_size = 1,
                 c_wr=0.50,
                 w0=30.0
                 ):
        super().__init__()
        self.c_wr = c_wr
        self.register_buffer('s_wr', torch.tensor(init_s_wr, dtype=torch.float32))

        self.embedding = nn.Sequential(
            nn.Linear(2, emb_size),
            Sine(w0),
            nn.Linear(emb_size, out_size),
            Sine(w0)
        )
        # 초기화
        self.embedding[0].apply(lambda m: siren_init(m, w0=w0, is_first=True))
        self.embedding[2].apply(lambda m: siren_init(m, w0=w0, is_first=False))

    
    def normalizer(self,wr):
        return normalize_winrate(wr,self.s_wr,self.c_wr)

    def forward(self, wr, pb):
        wr_norm = self.normalizer(wr)
        pb = torch.tensor(pb, dtype=torch.float32, device=wr_norm.device)
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
                 w0 =30,
                 ):
        super().__init__()
        self.register_buffer('s_wr', torch.tensor(init_s_wr, dtype=torch.float32))
        self.c_wr = c_wr
        
        self.register_buffer('s_game', torch.tensor(init_s_game, dtype=torch.float32))
        self.mim_game = mim_game
        
        self.normalize_wr = normalize_winrate
        self.normalize_ms = normalize_mastery

        self.embedding = nn.Sequential(
            nn.Linear(2, emb_size),
            Sine(w0),
            nn.Linear(emb_size, out_size),
            Sine(w0)
        )
        # 초기화
        self.embedding[0].apply(lambda m: siren_init(m, w0=w0, is_first=True))
        self.embedding[2].apply(lambda m: siren_init(m, w0=w0, is_first=False))
    
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
                 w0 = 30,
                 ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(n_list, emb_size),
            Sine(w0),
            nn.Linear(emb_size, out_size),
            Sine(w0)
        )
        # 초기화
        self.embedding[0].apply(lambda m: siren_init(m, w0=w0, is_first=True))
        self.embedding[2].apply(lambda m: siren_init(m, w0=w0, is_first=False))

    def forward(self,input_stack):
        output = self.embedding(input_stack)

        return output

class Encoder_position(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=8, w0=30):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Sine(w0),
            nn.Linear(hidden_dim, output_dim),
            Sine(w0)
        )
        # 초기화
        self.mlp[0].apply(lambda m: siren_init(m, w0=w0, is_first=True))
        self.mlp[2].apply(lambda m: siren_init(m, w0=w0, is_first=False))
        self.attn = nn.Linear(output_dim, 1)  # Attention score per position

    def forward(self,rank,lg,player,po):
        x = torch.cat([rank, lg, player,po], dim=1)
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
                 w0 = 30,
                 ):
        super().__init__()
        self.normalize = normalize_ELO
        self.mu = mu
        self.sigma = sigma
        self.embedding = nn.Sequential(
            nn.Linear(1, emb_size),
            Sine(w0),
            nn.Linear(emb_size, out_size),
            Sine(w0)
        )
        # 초기화
        self.embedding[0].apply(lambda m: siren_init(m, w0=w0, is_first=True))
        self.embedding[2].apply(lambda m: siren_init(m, w0=w0, is_first=False))

    def forward(self,ELO):
        ELO_nm = self.normalize(ELO,self.mu,self.sigma)
        output = self.embedding(ELO_nm)

        return output

class Encoder_po(nn.Module):
    def __init__(self,
                 #Param
                 init_s_wr=40.0,
                 emb_size = 4,
                 out_size = 1,
                 # Value
                 c_wr=0.50,
                 w0 = 30,
                 ):
        super().__init__()
        self.c_wr = c_wr
        self.register_buffer('s_wr', torch.tensor(init_s_wr, dtype=torch.float32))
        self.normalize = normalize_winrate
        self.embedding = nn.Sequential(
            nn.Linear(2, emb_size),
            Sine(w0),
            nn.Linear(emb_size, out_size),
            Sine(w0)
        )
        # 초기화
        self.embedding[0].apply(lambda m: siren_init(m, w0=w0, is_first=True))
        self.embedding[2].apply(lambda m: siren_init(m, w0=w0, is_first=False))

    def forward(self, wr, pb):
        wr_norm = self.normalize(wr,self.s_wr,self.c_wr)
        pb = torch.tensor(pb, dtype=torch.float32, device=wr_norm.device)
        enc_input = torch.stack([wr_norm,pb])
        output = self.embedding(enc_input)

        return output
    
class MLP(nn.Module):
    def __init__(self,
                 in_size = 16,
                 emb_size = 16,
                 out_size = 1,
                 w0=30.0
                 ):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_size, emb_size),
            Sine(w0),
            nn.Linear(emb_size, emb_size),
            Sine(w0),
            nn.Linear(emb_size, emb_size),
            Sine(w0),
            nn.Linear(emb_size, out_size),
            Sine(w0)
        )
        # 초기화
        self.embedding[0].apply(lambda m: siren_init(m, w0=w0, is_first=True))
        self.embedding[2].apply(lambda m: siren_init(m, w0=w0, is_first=False))
        self.embedding[4].apply(lambda m: siren_init(m, w0=w0, is_first=False))
        self.embedding[6].apply(lambda m: siren_init(m, w0=w0, is_first=False))

    def forward(self,input):
        output = self.embedding(input)

        return output