# _Module.py
import torch
import torch.nn as nn
import numpy as np

from _utils import *
from _Load import *
from _Config import regions, leagues, region_map

def _module_device(module: nn.Module):
    return next(module.parameters()).device if any(p.requires_grad for p in module.parameters()) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Loader_player(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.loader = load_player
        self.cache = {}  # (team, champ, pos_idx) -> tensor

    def forward(self, team, pb):
        device = _module_device(self.encoder)
        result = []
        for pos_idx in range(len(pb)):
            key = (team, pb[pos_idx], pos_idx)
            if key in self.cache:
                out = self.cache[key].to(device)
            else:
                data = self.loader(team, pb[pos_idx], pos_idx)
                # 안전한 텐서 변환 + 디바이스 맞추기
                game = data["game_gamer"]
                wr   = data["wr_gamer"]
                if not isinstance(game, torch.Tensor): game = torch.tensor(game, dtype=torch.float32, device=device)
                else: game = game.to(device)
                if not isinstance(wr, torch.Tensor): wr = torch.tensor(wr, dtype=torch.float32, device=device)
                else: wr = wr.to(device)
                out = self.encoder(game, wr)
                # CPU 텐서로 캐싱(메모리 절약), 사용 시 .to(device)
                self.cache[key] = out.detach().cpu()
            result.append(out)
        output = torch.stack(result)
        return output

class Loader_match(nn.Module):
    def loader_game(self, data_save):
        name_game = data_save["game_name"]
        parts = name_game.split("_")
        team_b = parts[2]
        team_r = parts[4]

        data_game = data_save["game_data"]
        pb_b_raw = data_game["blue_picks"]
        pb_r_raw = data_game["red_picks"]
        pb_b = replace_champion_names(pb_b_raw)
        pb_r = replace_champion_names(pb_r_raw)
        gold_diff = data_game['gold_diff'][1:]
        return {"B": team_b, "R": team_r, "pb_B": pb_b, "pb_R": pb_r, 'gold_diff': gold_diff}

    def __init__(self):
        super().__init__()
    def forward(self, data_match):
        return [self.loader_game(data_match[g]) for g in range(len(data_match))]
        # (기존 로직 그대로면 원래 코드 유지해도 됨)
        # ※ 위 한 줄은 기존 그대로 두셔도 됩니다. 여기서는 생략 가능.

class Loader_champ(nn.Module):
    def __init__(self, encoder, loader, reader, stacker):
        super().__init__()
        self.encoder = encoder
        self.loader = loader
        self.enc = reader
        self.stacker = stacker
        self.cache = {}  # (champ, pos_idx, tag) -> stacked tensor

    def forward(self, pb):
        device = _module_device(self.encoder)
        result = []
        for pos_idx in range(len(pb)):
            key = (pb[pos_idx], pos_idx, id(self.enc))  # reader별로 키 구분
            if key in self.cache:
                out_tensor = self.cache[key].to(device)
            else:
                data = self.loader(pb[pos_idx], pos_idx)
                out_dict = self.enc(data, self.encoder)  # 내부에서 encoder 호출
                out_tensor = self.stacker(out_dict)
                self.cache[key] = out_tensor.detach().cpu()
                out_tensor = out_tensor.to(device)
            result.append(out_tensor)
        output = torch.stack(result)
        return output

class Loader_po(nn.Module):
    def __init__(self, nm, loader, reader, stacker):
        super().__init__()
        self.nm = nm
        self.loader = loader
        self.enc = reader
        self.stacker = stacker
        self.cache = {}  # (round(time,3), champ, pos_idx) -> tensor

    def forward(self, time, pb):
        # 시간 연속값은 그대로 키로 쓰기 위험 -> 라운딩
        tkey = float(time.detach().cpu().item()) if isinstance(time, torch.Tensor) else float(time)
        tkey = round(tkey, 3)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        result = []
        for pos_idx in range(len(pb)):
            key = (tkey, pb[pos_idx], pos_idx)
            if key in self.cache:
                out_tensor = self.cache[key].to(device)
            else:
                data = self.loader(time, pb[pos_idx], pos_idx)
                out_dict = self.enc(data, self.nm)
                out_tensor = self.stacker(out_dict)
                self.cache[key] = out_tensor.detach().cpu()
                out_tensor = out_tensor.to(device)
            result.append(out_tensor)
        output = torch.stack(result)
        return output

def stacker_region(data_region):
    tensors = []
    for region in regions:
        k = f'result_rank_{region}'
        if k in data_region:
            tensors.extend(data_region[k])
    if tensors:
        out = torch.stack(tensors).view(-1)
    else:
        out = torch.empty(0)
    return out

def stacker_league(data_region):
    tensors = []
    for lg in leagues:
        k = f'result_lg_{lg}'
        if k in data_region:
            tensors.extend(data_region[k])
    if tensors:
        out = torch.stack(tensors).view(-1)
    else:
        out = torch.empty(0)
    return out

def reader_rank(data_champ, encoder):
    output = {}
    for region in regions:
        wr_key = f'wr_rank_{region}'
        pb_key = f'pb_rank_{region}'
        out_key = f'result_rank_{region}'
        if pb_key in data_champ and wr_key in data_champ:
            wr = data_champ[wr_key]; pb = data_champ[pb_key]
            # 안전한 텐서 변환은 encoder 안에서 처리되므로 그대로 전달
            output[out_key] = encoder(wr, pb)
    return output

def reader_lg(data_champ, encoder):
    output = {}
    for lg in leagues:
        wr_key = f'wr_lg_{lg}'
        pb_key = f'pb_lg_{lg}'
        out_key = f'result_lg_{lg}'
        if pb_key in data_champ and wr_key in data_champ:
            output[out_key] = encoder(data_champ[wr_key], data_champ[pb_key])
    return output

def reader_po(data_champ, nm):
    output = {}
    for region in regions:
        wr_key = f'po_{region}'
        out_key = f'result_rank_{region}'
        if wr_key in data_champ:
            v = data_champ[wr_key]
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32)
            output[out_key] = nm(v).reshape(-1, 1)
    return output
