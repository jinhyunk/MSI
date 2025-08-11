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
    
    bp = data_match[0]["pb_B"]
    idx_champ = Find_champion_idx(bp[0])
    print(idx_champ)
    po_graph = load_power(idx_champ,0)
    print(po_graph['po_rank_kr'])
    print(len(po_graph['po_rank_kr']))
    print(type(po_graph['po_rank_kr']))
    Model_po = Loader_po(Loader_po,reader_po,stacker_region)
    output = reader_po(po_graph)
    print(output)
    # graph_team_time_wr(output["po_kr"],0)
    # graph_team_time_wr(output["po_eu"],0)
    # graph_team_time_wr(output["po_na"],0)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data', default='./data', type=str, help='Data location')
    args = parser.parse_args()
    main(args)