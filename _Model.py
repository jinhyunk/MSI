import torch
import torch.nn as nn

from _Calc import *
from _utils import * 

d_enc = 4

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

class Embedding(nn.Module):
    def __init__(self, input_dim, emb_dim, out_dim, depth=2, w0=30.0):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(input_dim, emb_dim))
        layers.append(Sine(w0))

        for _ in range(depth - 1):
            layers.append(nn.Linear(emb_dim, emb_dim))
            layers.append(Sine(w0))

        # 출력층
        layers.append(nn.Linear(emb_dim, out_dim))
        layers.append(Sine(w0))

        self.net = nn.Sequential(*layers)

        # Siren 초기화 적용
        for idx, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                siren_init(layer, w0=w0, is_first=(idx == 0))

    def forward(self, x):
        return self.net(x)
    
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

        self.embedding = Embedding(2,emb_size,out_size,d_enc,w0)

    
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

        self.embedding = Embedding(2,emb_size,out_size,d_enc,w0)
    
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
        self.embedding = self.embedding = Embedding(n_list,emb_size,out_size,d_enc,w0)

    def forward(self,input_stack):
        output = self.embedding(input_stack)

        return output

class Encoder_position(nn.Module):
    def __init__(self, in_size=8, emb_size=16, out_size=8, w0=30):
        super().__init__()
        self.mlp = Embedding(in_size,emb_size,out_size,d_enc,w0)
        self.attn = nn.Linear(out_size, 1)  # Attention score per position

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
        self.embedding = Embedding(1,emb_size,out_size,d_enc,w0)

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
        self.embedding = Embedding(2,emb_size,out_size,d_enc,w0)

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
        self.embedding = Embedding(in_size,emb_size,out_size,8,w0)

    def forward(self,input):
        output = self.embedding(input)

        return output