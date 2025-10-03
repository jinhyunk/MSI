import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # GUI 비활성화
import matplotlib.pyplot as plt

from _Calc import calc_gold_wr
from _utils import replace_champion_names

# input() 자동 대체 (선택 0번 고정)
import builtins
builtins.input = lambda _: "0"


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
    
class MSIDataset(Dataset):
    def __init__(self, data_train, device):
        self.data_train = data_train
        self.device = device
        self.processed_data = self._preprocess_data()
    
    def _preprocess_data(self):
        processed_data = []
        Loader_model = Loader_match()
        
        print("데이터 전처리 중...")
        for idx_match in range(len(self.data_train)):
            data_raw = self.data_train[idx_match]["data_game"]
            data_idx = self.data_train[idx_match]["match_idx"]
            data_match = Loader_model(data_raw)
            
            for idx_game in range(len(data_match)):
                blue_team = data_match[idx_game]["B"]
                red_team = data_match[idx_game]["R"]
                bp_blue = data_match[idx_game]["pb_B"]
                bp_red = data_match[idx_game]["pb_R"]
                gold_diff = data_match[idx_game]["gold_diff"]
                
                total_time = len(gold_diff)
                for game_time in range(len(gold_diff)):
                    input_time = game_time / total_time
                    wr_blue_gt, wr_red_gt = calc_gold_wr(gold_diff[game_time], game_time)
                    
                    processed_data.append({
                        'input_time': input_time,
                        'blue_team': blue_team,
                        'red_team': red_team,
                        'bp_blue': bp_blue,
                        'bp_red': bp_red,
                        'data_idx': data_idx,
                        'wr_blue_gt': wr_blue_gt,
                        'wr_red_gt': wr_red_gt
                    })
        
        print(f"총 {len(processed_data)}개의 샘플이 준비되었습니다.")
        return processed_data
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        return {
            'input_time': torch.tensor(item['input_time'], dtype=torch.float32),
            'blue_team': item['blue_team'],
            'red_team': item['red_team'],
            'bp_blue': item['bp_blue'],
            'bp_red': item['bp_red'],
            'data_idx': item['data_idx'],
            'wr_blue_gt': torch.tensor(item['wr_blue_gt'], dtype=torch.float32),
            'wr_red_gt': torch.tensor(item['wr_red_gt'], dtype=torch.float32)
        }

def collate_fn(batch):
    input_times = torch.stack([item['input_time'] for item in batch])
    wr_blue_gts = torch.stack([item['wr_blue_gt'] for item in batch])
    wr_red_gts = torch.stack([item['wr_red_gt'] for item in batch])
    
    return {
        'input_times': input_times,
        'blue_teams': [item['blue_team'] for item in batch],
        'red_teams': [item['red_team'] for item in batch],
        'bp_blues': [item['bp_blue'] for item in batch],
        'bp_reds': [item['bp_red'] for item in batch],
        'data_indices': [item['data_idx'] for item in batch],
        'wr_blue_gts': wr_blue_gts,
        'wr_red_gts': wr_red_gts
    }
