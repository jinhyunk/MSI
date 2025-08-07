import argparse
from _Calc import * 
from _Load import * 
from _utils import * 
from _Plot import * 
from _Module import * 
from _Model import * 

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

    # Loader input : (ban pick 5)
    # Loader output : (value * region) dictionary

    Model_rank = Loader_champ(Encoder_champ(),load_rank,reader_rank)
    out = Model_rank(data_match[0]["pb_B"])
    # print(out[0])

    Model_rank = Loader_champ(Encoder_champ(),load_league,reader_lg)
    out = Model_rank(data_match[0]["pb_B"])
    # print(out[0])

    Model_player = Loader_player(Encoder_player())
    out = Model_player(data_match[0]["B"],data_match[0]["pb_B"])
    print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data', default='./data', type=str, help='Data location')
    args = parser.parse_args()
    main(args)