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
    
    data_train = load_match_train()
    print(len(data_train))
    #init Module
    module_champ_rank = Module_champ_tear(init_s_wr=40.0,init_w_score=0.7)
    module_champ_lg = Module_champ_tear(init_s_wr=10.0,init_w_score=0.1)
    
    for idx in range(0,len(data_train)):
        # Load data from train-set
        match = data_train[idx]
        tournament_idx = match["match_idx"]
        
        game_name = match["game_name"]
        parts = game_name.split("_")
        blue_team = parts[2] 
        red_team = parts[4]
        blue_ELO,red_ELO = Find_ELO(tournament_idx,blue_team,red_team)

        data_match = match['data']
        blue_pick = data_match["blue_picks"]
        red_pick = data_match["red_picks"]
        blue_pick = replace_champion_names(blue_pick)
        red_pick = replace_champion_names(red_pick)

        
        for pos_idx in range(0,5):
            data_blue_player = load_player(blue_team,blue_pick[pos_idx],pos_idx)
            # data_blue_champ = load_champion(blue_pick[pos_idx],pos_idx)
            
            mastery = calc_mastery(data_blue_player["game_gamer"],data_blue_player["wr_gamer"])
            print("Blue top champion shape : ",mastery)

            data_red_player = load_player(red_team,red_pick[pos_idx],pos_idx)
            # data_red_champ = load_champion(red_pick[pos_idx],pos_idx)

            
        print("Blue ELO : ",blue_ELO , "Red ELO : ",red_ELO ,)
        print("Game",tournament_idx,"Blue : ",blue_team , "Red : ", red_team)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data', default='./data', type=str, help='Data location')
    args = parser.parse_args()
    main(args)