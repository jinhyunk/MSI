import torch
import torch.nn as nn
import numpy as np

from _Calc import calc_mastery
from _utils import *
from _Load import *
from _Config import regions,leagues,region_map

class Loader_player(nn.Module):
    def __init__(self,
                 encoder,
                 ):
        super().__init__()
        self.encoder = encoder
        self.loader = load_player

    def forward(self,team,pb):
        result = []
        for pos_idx in range(0,len(pb)):
            data = self.loader(team,pb[pos_idx],pos_idx)
            out = self.encoder(data["game_gamer"],data["wr_gamer"])
            result.append(out)

        output = torch.stack(result) 
        return output

class Loader_match(nn.Module):
    def loader_game(self,data_save):
        name_game = data_save["game_name"]
        parts = name_game.split("_")
        team_b = parts[2] 
        team_r = parts[4]

        data_game = data_save["game_data"]
        
        pb_b_raw = data_game["blue_picks"]
        pb_r_raw = data_game["red_picks"]
        pb_b = replace_champion_names(pb_b_raw)
        pb_r = replace_champion_names(pb_r_raw)

        return {
            "B":team_b,
            "R":team_r,
            "pb_B":pb_b,
            "pb_R":pb_r
        }
        
    def __init__(self,):
        super().__init__()
    
    def forward(self, data_match):
        data_tot = []
        for i in range(0,len(data_match)):
            data_game = self.loader_game(data_match[i])
            data_tot.append(data_game)

        return data_tot 
    
class Loader_champ(nn.Module):
    def __init__(self,
                 encoder,
                 loader,
                 reader,
                 stacker,
                 ):
        super().__init__()
        self.encoder = encoder
        self.loader = loader
        self.enc = reader
        self.stacker = stacker

    def forward(self, pb):
        result = []
        for pos_idx in range(0,len(pb)):
            data = self.loader(pb[pos_idx],pos_idx)
            out_dict = self.enc(data,self.encoder)
            out_tensor = self.stacker(out_dict)  
            result.append(out_tensor)

        output = torch.stack(result)  # shape: (5, R, D)

        return output

def stacker_region(data_region):
    tensors = []

    for region in regions:
        stack_key = f'result_rank_{region}'
        
        if stack_key in data_region:
            tensors.extend(data_region[stack_key])
        
    if tensors:
        output = torch.stack(tensors).view(-1)  # (n,)
    else:
        output = torch.tensor([])  

    return output

def stacker_league(data_region):
    tensors = []

    for region in leagues:
        stack_key = f'result_lg_{region}'
        
        if stack_key in data_region:
            tensors.extend(data_region[stack_key])
        
    if tensors:
        output = torch.stack(tensors).view(-1)  # (n,)
    else:
        output = torch.tensor([])  

    return output

def reader_rank(data_champ,encoder):
    output = {}
    for region in regions:
        wr_key = f'wr_rank_{region}'
        pb_key = f'pb_rank_{region}'
        out_key = f'result_rank_{region}'
        if pb_key in data_champ and wr_key in data_champ:
            output[out_key] = encoder(data_champ[wr_key],data_champ[pb_key])
        
    return output

def reader_lg(data_champ,encoder):
    output = {}
    for region in leagues:
        wr_key = f'wr_lg_{region}'
        pb_key = f'pb_lg_{region}'
        out_key = f'result_lg_{region}'
        if pb_key in data_champ and wr_key in data_champ:
            output[out_key] = encoder(data_champ[wr_key],data_champ[pb_key])
        
    return output