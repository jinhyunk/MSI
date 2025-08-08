import argparse
from _Calc import * 
from _Load import * 
from _utils import * 
from _Plot import * 
from _Module import * 
from _Model import * 
from _Config import * 

def main(args):
    args = args
    #device = args.device
    path_data = args.data
    
    data_train = load_match_train()
    #init Module
    Model = Loader_match()
    # data_train ( match * (match_idx,[game * data])
    data_match = data_train[0]["data_game"]
    data_idx = data_train[0]["match_idx"]
    data_match = Model(data_match)
    # data_match ( game * (B,R,banpick) )
    print(data_match[0])
    ELO_B,ELO_R = Find_ELO(data_idx,data_match[0]["B"],data_match[0]["R"])
    print("Blue team",data_match[0]["B"],"ELO : ",ELO_B)
    # Loader input : (ban pick 5)
    # Loader output : (value * region) dictionary

    Model_rank = Loader_champ(Encoder_champ(),load_rank,reader_rank,stacker_region)
    out = Model_rank(data_match[0]["pb_B"])
    print(out.shape)
    print(out[0])
    # check = results_stack_region(out[0])
    # print(check)
    Model_sum = Encoder_region(len(regions),4)
    check_out_rank = Model_sum(out)
    print(check_out_rank.shape)

    print("______________________________")

    Model_rank = Loader_champ(Encoder_champ(),load_league,reader_lg,stacker_league)
    out = Model_rank(data_match[0]["pb_B"])
    print(out.shape)
    print(out[0])
    # check = results_stack_league(out[0])
    # print(check)
    Model_sum = Encoder_region(len(leagues),4)
    check_out_lg = Model_sum(out)
    print(check_out_lg.shape)

    print("______________________________")


    Model_player = Loader_player(Encoder_player())
    out_gamer = Model_player(data_match[0]["B"],data_match[0]["pb_B"])
    print(out_gamer.shape)

    Model_final = Encoder_position(8)
    out = Model_final(check_out_rank,check_out_lg,out_gamer)
    print(out.shape)

    Model_ELO = Encoder_ELO()
    ELO = torch.tensor([ELO_B], dtype=torch.float32)
    ELO = ELO.view(-1, 1)
    output = Model_ELO(ELO)
    print(output)
    print("Blue team",data_match[0]["B"],"ELO : ",ELO_B)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data', default='./data', type=str, help='Data location')
    args = parser.parse_args()
    main(args)