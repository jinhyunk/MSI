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

from _Calc import * 
from _Load import * 
from _utils import * 
from _Plot import * 
from _Module import * 
from _Model import * 
from _Config import * 
from _MSI_gpu import * 

# input() 자동 대체 (선택 0번 고정)
import builtins
builtins.input = lambda _: "0"

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

def main(args):
    # ===== Device 설정 =====
    device = args.device
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU 사용: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("CPU 사용")

    # ===== 모델 & 옵티마이저 =====
    Model = Model_MSI_Batch().to(device)
    total_params = sum(p.numel() for p in Model.parameters())
    trainable_params = sum(p.numel() for p in Model.parameters() if p.requires_grad)
    print(f"총 파라미터 수: {total_params:,}, 학습 가능 파라미터 수: {trainable_params:,}")

    optimizer = optim.Adam(Model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    criterion = nn.MSELoss()

    # ===== 데이터 로드 =====
    data_train = load_match_train()
    dataset = MSIDataset(data_train, device)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True
    )

    # ===== 학습 준비 =====
    iters = args.epochs
    loss_history = []
    best_loss = float('inf')
    start_time = time.time()

    print(f"\n===== 학습 시작 =====")
    print(f"Epochs: {iters}, Learning Rate: {args.learning_rate}, Batch Size: {args.batch_size}\n")

    # ===== 학습 루프 =====
    for epoch in range(iters):
        epoch_loss = 0.0
        batch_count = 0
        Model.train()

        for batch_data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{iters}"):
            input_times = batch_data['input_times'].to(device, non_blocking=True)
            wr_blue_gts = batch_data['wr_blue_gts'].to(device, non_blocking=True).view(-1, 1)
            wr_red_gts  = batch_data['wr_red_gts'].to(device, non_blocking=True).view(-1, 1)

            blue_teams   = batch_data['blue_teams']
            red_teams    = batch_data['red_teams']
            bp_blues     = batch_data['bp_blues']
            bp_reds      = batch_data['bp_reds']
            data_indices = batch_data['data_indices']

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                # 배치 단위 예측
                wr_blues = Model.forward_batch(input_times, blue_teams, bp_blues, data_indices)
                wr_reds  = Model.forward_batch(input_times, red_teams,  bp_reds,  data_indices)

                loss_blue = criterion(wr_blues, wr_blue_gts)
                loss_red  = criterion(wr_reds, wr_red_gts)
                total_batch_loss = loss_blue + loss_red

            # Mixed Precision Gradient 계산
            scaler.scale(total_batch_loss).backward()

            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(Model.parameters(), args.gradient_clip)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_batch_loss.item()
            batch_count += 1

        avg_epoch_loss = epoch_loss / batch_count
        loss_history.append(avg_epoch_loss)

        if args.use_scheduler:
            scheduler.step(avg_epoch_loss)

        # Best Model 저장
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            if args.save_best_model:
                torch.save(Model.state_dict(), args.best_model_path)

        print(f"[{epoch+1:03d}/{iters}] Loss: {avg_epoch_loss:.6f} | Best: {best_loss:.6f}")

    total_time = time.time() - start_time
    print("\n===== 학습 종료 =====")
    print(f"총 학습 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")

    # 손실 그래프 저장
    plt.plot(loss_history)
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("loss_history.png")
    print("손실 그래프 저장 완료: loss_history.png")

    if args.save_model:
        torch.save(Model.state_dict(), args.model_path)
        print(f"모델 저장 완료: {args.model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSI 모델 배치 학습 스크립트')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--epochs', default=300, type=int)  # 테스트 시 10으로 줄임
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--save_best_model', action='store_true')
    parser.add_argument('--model_path', default='./weight/model_1e_3/model_MSI_batch.pth', type=str)
    parser.add_argument('--best_model_path', default='./weight/model_1e_3/model_MSI_batch_best.pth', type=str)
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--gradient_clip', default=1.0, type=float)
    args = parser.parse_args()
    main(args)
