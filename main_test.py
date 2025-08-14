import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt

from _Calc import * 
from _Load import * 
from _utils import * 
from _Plot import * 
from _Module import * 
from _Model import * 
from _Config import * 
from _MSI_gpu import * 

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
    
def main(args):
    args = args
    device = args.device
    
    # GPU 사용 가능 여부 확인
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CPU 사용")
    
    #init Model
    Model = Model_MSI().to(device)
    Model.load_state_dict(torch.load(args.model_path, map_location=device))
    Model.eval()
    print(f"모델 로드 완료: {args.model_path}")
    
    # Data Loader
    data_test = load_match_test(path="./data/Game/")
    Loader_model = Loader_match()

    match_idx = data_test[0]['match_idx']
    match_data = data_test[0]['data_game']
    print(match_idx)
    data_match = Loader_model(match_data)
    for idx_game in range(len(data_match)):
        blue_team = data_match[idx_game]["B"]
        red_team = data_match[idx_game]["R"]
        bp_blue = data_match[idx_game]["pb_B"]
        bp_red = data_match[idx_game]["pb_R"]
        gold_diff = data_match[idx_game]["gold_diff"]

        total_time = len(gold_diff)
        wr_blue_tot = []
        wr_red_tot = []
        wr_blue_gt_tot = []
        wr_red_gt_tot = []

        for game_time in range(len(gold_diff)):
            input_time = game_time / total_time
            wr_blue = Model(input_time,blue_team,bp_blue,match_idx)
            wr_red = Model(input_time,red_team,bp_red,match_idx)
            wr_blue_gt, wr_red_gt = calc_gold_wr(gold_diff[game_time], game_time)
            
            wr_blue_tot.append(wr_blue.reshape(-1).detach().cpu().numpy())
            wr_red_tot.append(wr_red.reshape(-1).detach().cpu().numpy())
            wr_blue_gt_tot.append(wr_blue_gt)
            wr_red_gt_tot.append(wr_red_gt)

        graph_team_time_wr(wr_blue_tot)
        graph_team_time_wr(wr_red_tot)
        graph_team_time_wr(wr_blue_gt_tot)
        graph_team_time_wr(wr_red_gt_tot)
                    
    
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--model_path', default='./weight/model_1e_3/model_MSI_batch_best.pth', type=str, help='Path to save model')
    args = parser.parse_args()
    main(args)