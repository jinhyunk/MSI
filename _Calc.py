import numpy as np 

from _utils import * 
from _Load import * 
from math import exp

import numpy as np

def calc_mastery(game_gamer, wr_gamer, master=10, scale=0.2, w_gamer=0.9):
    def func_activation(game, master=10, scale=0.2, max_val=0.9):
        sig = 1 / (1 + np.exp(-scale * (game - master)))  # 0 ~ 1
        return sig * max_val
    
    nm_game = func_activation(game_gamer, master, scale)
    base_mastery = w_gamer * nm_game + (1.0 - w_gamer) * wr_gamer

    def bonus_signature(game, wr, min_game=50, min_wr=0.55, bonus_max=0.05):
        if game < min_game or wr < min_wr:
            return 0.0
        wr_adj = min((wr - min_wr) / (0.65 - min_wr), 1.0)
        return bonus_max * wr_adj

    bonus = bonus_signature(game_gamer, wr_gamer)

    return base_mastery + bonus

def calc_winrate(elo_team1, elo_team2):
    
    probability = 1 / (1 + math.pow(10, (elo_team2 - elo_team1) / 400))
    return probability , 1.0-probability

def calc_po_champ(po_kr, po_eu, po_na, weights):
    
    return (
        weights[0] * po_kr +
        weights[1] * po_eu +
        weights[2] * po_na
    )

def calc_po_compare(data_dict):
    def calc_po_region(po, wr_rank):
        return po - wr_rank
    regions = ['kr', 'eu', 'na']
    
    for region in regions:
        po_key = f'po_{region}'
        wr_key = f'wr_rank_{region}'
        comp_key = f'po_compare_{region}'
        
        if po_key in data_dict and wr_key in data_dict:
            data_dict[comp_key] = calc_po_region(data_dict[po_key], data_dict[wr_key])
            
    return data_dict

def calc_po_compare(po,wr_rank):
    return po - wr_rank 
    
def calc_wr_team(ELO1,ELO2,gamer):
    wr_B, wr_R = calc_winrate(ELO1,ELO2)
