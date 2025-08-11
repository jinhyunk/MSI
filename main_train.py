import argparse
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
    #device = args.device
    path_data = args.data
    
    #init Model
    Model = Model_MSI()
    
    # Data Loader
    data_train = load_match_train()
    Loader_model = Loader_match()

    #Training settings
    iters = 1000

    for iters in range(0,iters):
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
                    wr_blue = Model(input_time,blue_team,bp_blue,data_idx) 
                    wr_red = Model(input_time,red_team,bp_red,data_idx) 
                    gt_blue,gt_red = calc_gold_wr(gold_diff[game_time],game_time)
                    # print("Blue team 승률 : ", wr_blue, "Blue team 승률 GT : ",gt_blue)
                    # print("Red team 승률 : ", wr_red, "Red team 승률 GT : ",gt_red)


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
    #parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data', default='./data', type=str, help='Data location')
    args = parser.parse_args()
    main(args)