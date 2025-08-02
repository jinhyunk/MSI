import argparse
from _Calc import * 
from _Load import * 
from _utils import * 
from _Plot import * 
from _Module import * 

def main(args):
    args = args
    #device = args.device
    path_data = args.data
    idx_match = args.idx
    
    url_match = "https://gol.gg/game/stats/"+str(idx_match)+"/page-game/"

    data_match = load_match(url_match)

    data_gold_diff = data_match["gold_diff"]
    blue_name = data_match["B"]
    red_name = data_match["R"]
    blue_pick = data_match["blue_picks"]
    red_pick = data_match["red_picks"]
    blue_pick = replace_champion_names(blue_pick)
    red_pick = replace_champion_names(red_pick)

    #init Module
    module_champ_rank = Module_champ_tear(init_s_wr=40.0,init_w_score=0.7)
    module_champ_lg = Module_champ_tear(init_s_wr=10.0,init_w_score=0.1)
    
    # Load data
    data_player = load_player("GEN",blue_pick[0],0)
    data_champ = load_champion(red_pick[2],2)
    
    # Calc data
    data_po_time = calc_po_compare(data_champ)

    #Example 
    example = module_champ_rank(data_champ["wr_rank_kr"],data_champ["pb_rank_kr"])
    print(example)
    example2 = module_champ_lg(data_champ["wr_lg_LCK"],data_champ["pb_lg_LCK"])
    print(example2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data', default='./data', type=str, help='Data location')
    parser.add_argument('--idx', default='69323', type=str, help='gol.gg match idx 69322/69314/69297')
    args = parser.parse_args()
    main(args)