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
    ELO = torch.tensor([ELO_B], dtype=torch.float32)
    ELO = ELO.view(-1, 1)
    
    Model = MSI()
    out = Model(data_match[0]["B"],ELO,data_match[0]["pb_B"])
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data', default='./data', type=str, help='Data location')
    args = parser.parse_args()
    main(args)