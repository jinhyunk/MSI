import torch
import torch.nn as nn

from _Model import * 
from _Module import * 

class Model_MSI(nn.Module):
    def __init__(self,
                 s_rank = 40.0,
                 s_lg = 10.0,
                 s_player_wr = 10.0,
                 s_player_game = 0.2,

                 emb_size_enc = 4,
                 emb_size_sum = 2,
                 c_wr = 0.50,
                 mim_game = 5,
                 ELO_mu = 1500,
                 ELO_sigma = 200,

                 ):
        super().__init__()
        self.enc_rank = Encoder_champ(init_s_wr=s_rank,
                                      emb_size=emb_size_enc,out_size=1,
                                      c_wr=c_wr)
        self.enc_lg = Encoder_champ(init_s_wr=s_lg,
                                      emb_size=emb_size_enc,out_size=1,
                                      c_wr=c_wr)
        self.enc_player = Encoder_player(init_s_wr=s_player_wr,init_s_game=s_player_game,
                                         emb_size=emb_size_enc,out_size=emb_size_sum,
                                         c_wr=c_wr,mim_game=mim_game)

        self.model_rank = Loader_champ(self.enc_rank,
                                       load_rank,reader_rank,
                                       stacker_region)
        self.model_lg = Loader_champ(self.enc_lg,
                                     load_league,reader_lg,
                                     stacker_league)

        self.model_player = Loader_player(self.enc_player)

        self.model_po = Loader_po(self.enc_rank.normalizer,
                                  load_power,reader_po,stacker_region)

        self.sum_region = Encoder_region(len(regions),
                                       emb_size_enc,emb_size_sum)
        self.sum_lg = Encoder_region(len(leagues),
                                     emb_size_enc,emb_size_sum)
        
        self.enc_position = Encoder_position(input_dim=4*emb_size_sum,
                                             hidden_dim=8*emb_size_sum,
                                             output_dim=4*emb_size_sum)
        
        self.enc_ELO = Encoder_ELO(emb_size=8*emb_size_sum,
                                   out_size=4*emb_size_sum,
                                   mu=ELO_mu,sigma=ELO_sigma)
        
        self.MLP = MLP(in_size = 8 * emb_size_sum,
                       emb_size = 16 * emb_size_sum,
                       out_size = 1)
        
        self.ELO = Find_ELO
        
    def forward_sep(self,input,loader,encoder):
        out_enc = loader(input)
        out_final = encoder(out_enc)
        return out_final
    
    def forward_po(self,time,pb,loader,encoder):
        out_enc = loader(time,pb)
        out_final = encoder(out_enc)
        return out_final
        

    def forward(self,time,team,pb,idx_match):
        result_rank = self.forward_sep(pb,self.model_rank,self.sum_region)
        result_lg = self.forward_sep(pb,self.model_lg,self.sum_lg)
        result_player = self.model_player(team,pb)
        result_po = self.forward_po(time,pb,self.model_po,self.sum_region)
        
        result_champion = self.enc_position(result_rank,result_lg,result_player,result_po)

        ELO = self.ELO(idx_match,team)
        result_ELO = self.enc_ELO(ELO)

        out = torch.cat([result_champion,result_ELO],dim=1)
        result = self.MLP(out)

        return result
    