import numpy as np 

from _utils import * 
from _Load import * 

def calc_wr_total(team,pick_data,region=0):

    results = []
    for i in range(0,len(pick_data)):
        champ = pick_data[i]
        player_name = Find_player(team,i)
        game,wr_player = load_player(champ,team,player_name)
        pickban_rank,wr_rank = load_rank(champ,i)
        pickban_league,wr_league = load_league(champ,i)
        
        results.append({
            "game": game,
            "wr_player": wr_player,
            "pickban_rank": pickban_rank,
            "wr_rank": wr_rank,
            "pickban_league": pickban_league,
            "wr_league": wr_league
        })

    return results

def calc_wr_time_avg(pick_data):
    total_power = None

    for i in range(0,len(pick_data)):
        champs_name = pick_data[i]
        champs_idx = Find_champion_idx(champs_name)
        if champs_idx == None:
            print(f"❌ '{champs_name}'에 해당하는 데이터 인덱스가 존재하지 않습니다.")
        
        power_graph_kr = load_power(champs_idx,i,0)
        power_graph_eu = load_power(champs_idx,i,1)
        power_graph_na = load_power(champs_idx,i,3)
        if total_power is None:
            total_power = np.zeros_like(power_graph_kr)
        power_graph = (power_graph_kr + power_graph_eu + power_graph_na) / 3.0
        total_power = total_power + power_graph
        
    total_power = total_power/ 5.0
    
    return total_power

def calc_wr_time(pick_data,region=0):
    total_power = None

    for i in range(0,len(pick_data)):
        champs_name = pick_data[i]
        champs_idx = Find_champion_idx(champs_name)
        if champs_idx == None:
            print(f"❌ '{champs_name}'에 해당하는 데이터 인덱스가 존재하지 않습니다.")
        
        power_graph = load_power(champs_idx,i,region)
        if total_power is None:
            total_power = np.zeros_like(power_graph)
        total_power = total_power + power_graph
        
    total_power = total_power/ 5.0
    
    return total_power