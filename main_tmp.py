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
    w_po = init_weights(0)

    # for i in range(0,len(blue_pick)):
    #     champ_pick = blue_pick[i]
    #     data_champ = load_champion(blue_name,champ_pick,i)
    #     ms_gamer = calc_mastery(data_champ["game_gamer"],data_champ["wr_gamer"])
    #     po_ovr = calc_po_champ(data["po_kr"],data["po_eu"],data["po_na"],w_po)
    
    data = load_champion(blue_name,blue_pick[3],3)
    po_region = calc_po_compare(data["po_kr"],data["wr_rank_kr"])
    graph_team_time_wr(po_region,0)

    # graph_team_time_wr(data["po_eu"],data["wr_rank_eu"])
    # graph_team_time_wr(data["po_na"],data["wr_rank_na"])
    # graph_team_time_wr(data["po_eu"])
    # graph_team_time_wr(data["po_na"])
    # graph_team_time_wr(po_weight)
    # graph_compare_time_wr(data["po_kr"],po_weight)
    # print("Gamers game : ",data["game_gamer"])
    # print("Gamers_win rate : ",data["wr_gamer"])
    # print("lg_win rate : ",data["wr_lg"])
    # print("rank_win rate : ",data["wr_rank"])
    # nm_gamer = calc_mastery(data["game_gamer"],data["wr_gamer"])
    # print(data["Gamer"] + "의 " + blue_pick[0] + " 숙련도는 : " + str(nm_gamer) + "입니다.")
    
    # game_sc,wr_sc = load_player("K'Sante","GEN","Kiin")
    # nm_gamer_sc = calc_mastery(game_sc,wr_sc)
    # print(data["Gamer"] + "의 " + "K'Sante" + " 숙련도는 : " + str(nm_gamer_sc) + "입니다.")
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data', default='./data', type=str, help='Data location')
    parser.add_argument('--idx', default='69314', type=str, help='gol.gg match idx 69322/69314/69297')
    args = parser.parse_args()
    main(args)