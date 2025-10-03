import torch
import torch.nn as nn
from _utils import *
from _Load import *
from _Config import regions, leagues

def _get_device(module: nn.Module):
    """모듈의 디바이스를 안정적으로 찾는 함수"""
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Loader_player(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.loader = load_player
        self.cache = {}

    def forward(self, team, pb):
        device = _get_device(self.encoder)
        outputs = []
        for pos_idx, champ in enumerate(pb):
            key = (team, champ, pos_idx)
            if key in self.cache:
                out = self.cache[key].to(device)
            else:
                data = self.loader(team, champ, pos_idx)
                game = torch.tensor(data["game_gamer"], dtype=torch.float32, device=device)
                wr = torch.tensor(data["wr_gamer"], dtype=torch.float32, device=device)
                out = self.encoder(game, wr)
                # [수정] .detach()를 추가하여 계산 그래프로부터 분리 후 저장
                self.cache[key] = out.detach().cpu()
            outputs.append(out)
        return torch.stack(outputs)

class Loader_match(nn.Module):
    # ... (이 클래스는 수정할 필요 없습니다) ...
    def __init__(self):
        super().__init__()

    def forward(self, data_match):
        processed_games = []
        for game_data in data_match:
            name_game = game_data["game_name"]
            parts = name_game.split("_")
            team_b, team_r = parts[2], parts[4]

            game_details = game_data["game_data"]
            processed_games.append({
                "B": team_b, "R": team_r,
                "pb_B": replace_champion_names(game_details["blue_picks"]),
                "pb_R": replace_champion_names(game_details["red_picks"]),
                'gold_diff': game_details['gold_diff'][1:]
            })
        return processed_games

class Loader_champ(nn.Module):
    def __init__(self, encoder, loader, reader, stacker):
        super().__init__()
        self.encoder = encoder
        self.loader = loader
        self.reader = reader
        self.stacker = stacker
        self.cache = {}

    def forward(self, pb):
        device = _get_device(self.encoder)
        results = []
        for pos_idx, champ in enumerate(pb):
            key = (champ, pos_idx, self.loader.__name__)
            if key in self.cache:
                out_tensor = self.cache[key].to(device)
            else:
                data = self.loader(champ, pos_idx)
                out_dict = self.reader(data, self.encoder)
                out_tensor_cpu = self.stacker(out_dict)
                # [수정] .detach()를 추가하여 계산 그래프로부터 분리 후 저장
                self.cache[key] = out_tensor_cpu.detach().cpu()
                out_tensor = out_tensor_cpu.to(device)
            results.append(out_tensor)
        return torch.stack(results)

class Loader_po(nn.Module):
    def __init__(self, normalizer, loader, reader, stacker):
        super().__init__()
        self.normalizer = normalizer
        self.loader = loader
        self.reader = reader
        self.stacker = stacker
        self.cache = {}

    def forward(self, time, pb):
        time_key = round(time.item() if isinstance(time, torch.Tensor) else float(time), 3)
        device = _get_device(self)
        results = []
        for pos_idx, champ in enumerate(pb):
            key = (time_key, champ, pos_idx)
            if key in self.cache:
                out_tensor = self.cache[key].to(device)
            else:
                data = self.loader(time, champ, pos_idx)
                out_dict = self.reader(data, self.normalizer)
                out_tensor_cpu = self.stacker(out_dict)
                # [수정] .detach()를 추가하여 계산 그래프로부터 분리 후 저장
                self.cache[key] = out_tensor_cpu.detach().cpu()
                out_tensor = out_tensor_cpu.to(device)
            results.append(out_tensor)
        return torch.stack(results)

def stacker_region(data_region):
    tensors = [data_region[f'result_rank_{region}'] for region in regions if f'result_rank_{region}' in data_region]
    return torch.cat(tensors, dim=0) if tensors else torch.empty(0)

def stacker_league(data_league):
    tensors = [data_league[f'result_lg_{league}'] for league in leagues if f'result_lg_{league}' in data_league]
    return torch.cat(tensors, dim=0) if tensors else torch.empty(0)

def reader_rank(data_champ, encoder):
    output = {}
    for region in regions:
        wr_key, pb_key = f'wr_rank_{region}', f'pb_rank_{region}'
        if wr_key in data_champ:
            output[f'result_rank_{region}'] = encoder(data_champ[wr_key], data_champ[pb_key])
    return output

def reader_lg(data_champ, encoder):
    output = {}
    for league in leagues:
        wr_key, pb_key = f'wr_lg_{league}', f'pb_lg_{league}'
        if wr_key in data_champ:
            output[f'result_lg_{league}'] = encoder(data_champ[wr_key], data_champ[pb_key])
    return output

def reader_po(data_champ, normalizer):
    output = {}
    for region in regions:
        wr_key = f'po_{region}'
        if wr_key in data_champ:
            wr_tensor = torch.tensor(data_champ[wr_key], dtype=torch.float32)
            output[f'result_rank_{region}'] = normalizer(wr_tensor).reshape(-1, 1)
    return output