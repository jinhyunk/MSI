import torch
import torch.nn as nn
from _Module import *
from _Model import * 
from _Load import * 

class Model_MSI(nn.Module):
    def __init__(self,
                 s_rank=40.0, s_lg=10.0, s_player_wr=10.0, s_player_game=0.2,
                 emb_size_enc=4, emb_size_sum=2, c_wr=0.50, mim_game=5,
                 ELO_mu=1500, ELO_sigma=200):
        super().__init__()
        
        # Encoders
        self.enc_rank = Encoder_champ(init_s_wr=s_rank, emb_size=emb_size_enc, out_size=1, c_wr=c_wr)
        self.enc_lg = Encoder_champ(init_s_wr=s_lg, emb_size=emb_size_enc, out_size=1, c_wr=c_wr)
        self.enc_player = Encoder_player(init_s_wr=s_player_wr, init_s_game=s_player_game,
                                         emb_size=emb_size_enc, out_size=emb_size_sum,
                                         c_wr=c_wr, mim_game=mim_game)
        
        # Loaders
        self.model_rank = Loader_champ(self.enc_rank, load_rank, reader_rank, stacker_region)
        self.model_lg = Loader_champ(self.enc_lg, load_league, reader_lg, stacker_league)
        self.model_player = Loader_player(self.enc_player)
        self.model_po = Loader_po(self.enc_rank.normalizer, load_power, reader_po, stacker_region)
        
        # Summarizers
        self.sum_region = Encoder_region(len(regions), emb_size_enc, emb_size_sum)
        self.sum_lg = Encoder_region(len(leagues), emb_size_enc, emb_size_sum)
        
        # Positional and ELO Encoders
        self.enc_position = Encoder_position(in_size=4 * emb_size_sum,
                                             emb_size=8 * emb_size_sum,
                                             out_size=4 * emb_size_sum)
        self.enc_ELO = Encoder_ELO(emb_size=8 * emb_size_sum, out_size=4 * emb_size_sum,
                                   mu=ELO_mu, sigma=ELO_sigma)
        
        # Final MLP
        self.MLP = MLP(in_size=8 * emb_size_sum, emb_size=16 * emb_size_sum, out_size=1)
        
        self.ELO = Find_ELO

    def forward_single(self, time, team, pb, idx_match):
        """단일 샘플에 대한 forward 연산"""
        result_rank = self.sum_region(self.model_rank(pb))
        result_lg = self.sum_lg(self.model_lg(pb))
        result_player = self.model_player(team, pb)
        result_po = self.sum_region(self.model_po(time, pb).squeeze(-1))
        
        result_champion = self.enc_position(result_rank, result_lg, result_player, result_po)
        
        # [수정] Find_ELO가 float 값을 반환하므로, 이를 올바른 모양의 텐서로 만듭니다.
        elo_val = self.ELO(idx_match, team)
        elo_tensor = torch.tensor([[elo_val]], dtype=torch.float32, device=result_champion.device)
        
        result_ELO = self.enc_ELO(elo_tensor)
        
        # 이제 result_champion과 result_ELO 모두 (1, 8) 형태의 2D 텐서가 됩니다.
        out = torch.cat([result_champion, result_ELO], dim=1)
        return self.MLP(out)

    def forward_batch(self, times, teams, pbs, idx_matches):
        """배치 처리를 위한 forward 함수"""
        batch_size = len(teams)
        device = next(self.parameters()).device
        
        # 결과를 저장할 리스트
        results = []
        
        for i in range(batch_size):
            # forward_single을 호출하여 각 샘플을 처리
            result = self.forward_single(times[i], teams[i], pbs[i], idx_matches[i])
            results.append(result)
        
        # 결과 리스트를 하나의 텐서로 결합
        return torch.cat(results, dim=0)

    def forward(self, time, team, pb, idx_match):
        """기존 호환성을 위한 forward 함수"""
        return self.forward_single(time, team, pb, idx_match)