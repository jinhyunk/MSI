import numpy as np 

from _utils import * 
from _Load import * 
from _Config import leagues,regions,region_map
from math import exp
from scipy.interpolate import UnivariateSpline

import numpy as np
import torch 
import torch.nn as nn

def calc_ELO_wr(elo_team1, elo_team2):
    
    probability = 1 / (1 + math.pow(10, (elo_team2 - elo_team1) / 400))
    return probability , 1.0-probability

def calc_gold_wr(gold_diff, time_min, base_scale=750.0, alpha=0.04, k=2.0):
    """
    gold_diff: blue - red (정수, 음수 가능)
    time_min: 게임 시간 (분)
    base_scale: 시간=0일 때의 스케일(골드 단위)
    alpha: time에 따른 스케일 증가율 (클수록 후반 영향 감소 폭이 커짐)
    k: 로지스틱 기울기
    """
    
    scale = base_scale * (1 + alpha * time_min)
    z = k * (gold_diff / scale)
    p_blue = 1.0 / (1.0 + math.exp(-z))
    return p_blue, 1.0 - p_blue

def calc_wr_gold(win_prob, time_min, base_scale=750.0, alpha=0.04, k=2.0):
    if win_prob <= 0.0: return float('-inf')
    if win_prob >= 1.0: return float('inf')
    scale = base_scale * (1 + alpha * time_min)
    logit = math.log(win_prob / (1.0 - win_prob))
    gold_diff = (logit / k) * scale
    return gold_diff

def normalize_mastery(game, s_game=0.2 , mim_game=5):
        return 1 / (1 + torch.exp(-s_game * (game - mim_game)))  # 0 ~ 1

def normalize_winrate(wr,s_wr=40.0,c_wr=0.50):
        if isinstance(wr, torch.Tensor):
            wr_tensor = wr
        else:
            wr_tensor = torch.tensor(wr, dtype=torch.float32)
        return 1 / (1 + torch.exp(-s_wr * (wr_tensor - c_wr)))

def normalize_pickban(pb ,s_pb_1=25.0, s_pb_2=5.0,
                       c_pb_1=0.20,c_pb_2=0.80,
                       w_pb=0.55):
    if isinstance(pb, torch.Tensor):
        pb_tensor = pb
    else:
        pb_tensor = torch.tensor(pb, dtype=torch.float32)
    s1 = 1 / (1 + torch.exp(-s_pb_1 * (pb_tensor - c_pb_1)))
    s2 = 1 / (1 + torch.exp(-s_pb_2 * (pb_tensor - c_pb_2)))
    return w_pb * s1 + (1 - w_pb) * s2

def normalize_ELO(ELO, mu=1500, sigma=200):
    return 1 / (1 + torch.exp(-(ELO - mu) / sigma))

