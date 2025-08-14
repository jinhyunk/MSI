import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from _Calc import * 
from _Load import * 
from _utils import * 
from _Plot import * 
from _Module import * 
from _Model import * 
from _Config import * 
from _MSI import * 

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
    
    # Optimizer 설정
    optimizer = optim.Adam(Model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 손실함수 설정
    criterion = nn.MSELoss()
    
    # Data Loader
    data_train = load_match_train()
    Loader_model = Loader_match()

    #Training settings
    iters = args.epochs
    total_loss = 0.0
    num_batches = 0

    print(f"학습 시작: {iters} 에포크, 학습률: {args.learning_rate}")
    
    for epoch in range(0, iters):
        epoch_loss = 0.0
        batch_count = 0
        for idx_match in range(0,len(data_train)):
            # Load match data
            data_raw = data_train[idx_match]["data_game"]
            data_idx = data_train[idx_match]["match_idx"]
            data_match = Loader_model(data_raw)
            print("Now play match : ", data_idx)
            for idx_game in range(0,len(data_match)):
                # Load Game data
                blue_team = data_match[idx_game]["B"]
                red_team = data_match[idx_game]["R"]
                bp_blue = data_match[idx_game]["pb_B"]
                bp_red = data_match[idx_game]["pb_R"]

                gold_diff = data_match[idx_game]["gold_diff"]
                total_time = len(gold_diff) 
                for game_time in range(0,len(gold_diff)):
                    input_time = game_time / total_time
                    
                    # 모델 예측
                    wr_blue = Model(input_time,blue_team,bp_blue,data_idx) 
                    wr_red = Model(input_time,red_team,bp_red,data_idx) 
                    
                    # Ground truth 계산 (실제 골드 차이로부터 승률 계산)
                    wr_blue_gt,wr_red_gt = calc_gold_wr(gold_diff[game_time],game_time)
                    
                    # Ground truth를 텐서로 변환하고 device로 이동 (모델 출력과 같은 크기로)
                    gt_blue_tensor = torch.tensor([[wr_blue_gt]], dtype=torch.float32).to(device)
                    gt_red_tensor = torch.tensor([[wr_red_gt]], dtype=torch.float32).to(device)
                    
                    # 손실 계산 (모델 출력을 직접 사용)
                    loss_blue = criterion(wr_blue, gt_blue_tensor)
                    loss_red = criterion(wr_red, gt_red_tensor)
                    total_batch_loss = loss_blue + loss_red
                    
                    # 역전파
                    optimizer.zero_grad()
                    total_batch_loss.backward()
                    optimizer.step()
                    
                    epoch_loss += total_batch_loss.item()
                    batch_count += 1
                    
                                        # print("Blue team 승률 : ", wr_blue.item(), "Blue team 승률 GT : ",gt_blue)
                    # print("Red team 승률 : ", wr_red.item(), "Red team 승률 GT : ",gt_red)
        
        # 에포크 종료 후 평균 손실 출력
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        total_loss += avg_epoch_loss
        num_batches += 1
        
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch [{epoch+1}/{iters}], 평균 손실: {avg_epoch_loss:.6f}")
    
    # 전체 학습 완료 후 평균 손실 출력
    final_avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"학습 완료! 전체 평균 손실: {final_avg_loss:.6f}")
    
    # 모델 저장
    if args.save_model:
        torch.save(Model.state_dict(), args.model_path)
        print(f"모델이 {args.model_path}에 저장되었습니다.")


    # # data_match ( game * (B,R,banpick) )
    # print(data_match[0])
    # ELO_B= Find_ELO(data_idx,data_match[0]["B"])
    # print("Blue team",data_match[0]["B"],"ELO : ",ELO_B)
    # ELO = torch.tensor([ELO_B], dtype=torch.float32)
    # ELO = ELO.view(-1, 1)
    
    # Model = Model_MSI()
    # out = Model(0.2,data_match[0]["B"],data_match[0]["pb_B"],data_idx)
    # print(out.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of training epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay for optimizer')
    parser.add_argument('--print_every', default=100, type=int, help='Print loss every N epochs')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    parser.add_argument('--model_path', default='./model_weight/model_MSI.pth', type=str, help='Path to save model')
    args = parser.parse_args()
    main(args)