# config.py
import torch

# --- 기본 설정 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU_DEVICE = torch.device('cpu')

# --- 데이터 경로 ---
DATA_PATH = "./data/Game/"
WEIGHT_SAVE_PATH = "./weight/refactored/"
BEST_MODEL_PATH = f"{WEIGHT_SAVE_PATH}best_model.pth"
FINAL_MODEL_PATH = f"{WEIGHT_SAVE_PATH}final_model.pth"

# --- 데이터 로딩 설정 ---
REGIONS = ['kr', 'eu', 'na']
LEAGUES = ['LCK', 'LPL', 'LTA_N', 'LTA_S']
REGION_MAP = {'0': 'kr', '1': 'eu', '3': 'na'}

# --- 모델 하이퍼파라미터 ---
MODEL_PARAMS = {
    's_rank': 40.0,
    's_lg': 10.0,
    's_player_wr': 10.0,
    's_player_game': 0.2,
    'emb_size_enc': 4,
    'emb_size_sum': 2,
    'c_wr': 0.50,
    'mim_game': 5,
    'elo_mu': 1500,
    'elo_sigma': 200,
}

# --- 학습 하이퍼파라미터 ---
TRAIN_PARAMS = {
    'epochs': 300,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 256,
    'num_workers': 4,
    'use_scheduler': True,
    'gradient_clip': 1.0,
}