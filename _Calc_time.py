import numpy as np 

from _utils import * 
from _Load_match import * 
from _Load_lolps import * 


def Calc_time_pick(data_path,match_idx,game_name):
    
    data = load_match(data_path,match_idx,game_name)
    
    checker = "blue_picks"
    pick_data = data[checker]
    total_power = None

    for i in range(0,len(pick_data)):
        champs_name = pick_data[i]
        champs_idx = Find_champion_idx(champs_name)
        if champs_idx == None:
            print("Error w. champs_idx")

        power_graph = load_power(champs_idx,i)
        power_float = np.array(list(map(float, power_graph))) 
        if total_power is None:
            total_power = np.zeros_like(power_float) 

        total_power = total_power + power_float
        
    total_power = total_power/ 5.0
    
    blue_total_poewr = total_power

    checker = "red_picks"
    pick_data = data[checker]
    total_power = None

    for i in range(0,len(pick_data)):
        champs_name = pick_data[i]
        champs_idx = Find_champion_idx(champs_name)
        if champs_idx == None:
            print("Error w. champs_idx")

        power_graph = load_power(champs_idx,i)
        power_float = np.array(list(map(float, power_graph))) 
        if total_power is None:
            total_power = np.zeros_like(power_float) 

        total_power = total_power + power_float
        
    total_power = total_power/ 5.0
    
    red_total_poewr = total_power
    
    return blue_total_poewr,red_total_poewr