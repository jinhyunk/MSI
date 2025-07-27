import argparse
from _Calc import * 
from _Load import * 
from _utils import * 
from _Plot import * 

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
    data = load_champion("GEN",blue_pick[0],0)
    print(data.keys())
    w_po = init_weights(0)
    
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data', default='./data', type=str, help='Data location')
    parser.add_argument('--idx', default='69314', type=str, help='gol.gg match idx 69322/69314/69297')
    args = parser.parse_args()
    main(args)