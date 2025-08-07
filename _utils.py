import json
import math
import pandas as pd 
import numpy as np 

from _Config import PLAYER_CACHE

def Find_champion_idx(name_us, file_path="./json/champions.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for champ in data:
        if champ["nameUs"].lower() == name_us.lower():
            return champ["championId"]
    
    return None  # 찾지 못했을 경우

def Find_player_test(team_name, pos_idx, file_path="./json/roster.json"):
    position_names = {0: "Top", 1: "Jungle", 2: "Mid", 3: "ADC", 4: "Support"}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    team_found = False
    selected_players = []

    for team in data:
        if team["team"] == team_name:
            team_found = True
            for player in team["players"]:
                if player["position"] == pos_idx:
                    selected_players.append(player["name"])
            break

    if not team_found:
        print(f"Team '{team_name}' does not exist in the data.")
        return None

    if not selected_players:
        print(f"No players found in position '{position_names.get(pos_idx, 'Unknown')}' for team '{team_name}'.")
        return None

    if len(selected_players) == 1:
        return selected_players[0]

    print(f"Multiple players found for position '{position_names.get(pos_idx, 'Unknown')}' in team '{team_name}':")
    for i, name in enumerate(selected_players):
        print(f"{i}. {name}")

    while True:
        choice = input("Please enter the number of the player you want to select: ")
        if choice.isdigit():
            choice = int(choice)
            if 0 <= choice < len(selected_players):
                return selected_players[choice]
        print("Invalid input. Please try again.")

def Find_player(team_name, pos_idx, file_path="./json/roster.json"):
    
    position_names = {0: "Top", 1: "Jungle", 2: "Mid", 3: "ADC", 4: "Support"}
    key = (team_name, pos_idx)

    # 이미 선택된 플레이어가 있으면 캐시에서 반환
    if key in PLAYER_CACHE:
        return PLAYER_CACHE[key]

    # 파일 로드
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    team_found = False
    selected_players = []

    for team in data:
        if team["team"] == team_name:
            team_found = True
            for player in team["players"]:
                if player["position"] == pos_idx:
                    selected_players.append(player["name"])
            break

    if not team_found:
        print(f"Team '{team_name}' does not exist in the data.")
        return None

    if not selected_players:
        print(f"No players found in position '{position_names.get(pos_idx, 'Unknown')}' for team '{team_name}'.")
        return None

    if len(selected_players) == 1:
        PLAYER_CACHE[key] = selected_players[0]
        return selected_players[0]

    # 플레이어 선택 - 최초 1회
    print(f"Multiple players found for position '{position_names.get(pos_idx, 'Unknown')}' in team '{team_name}':")
    for i, name in enumerate(selected_players):
        print(f"{i}. {name}")

    while True:
        choice = input("Please enter the number of the player you want to select: ")
        if choice.isdigit():
            choice = int(choice)
            if 0 <= choice < len(selected_players):
                PLAYER_CACHE[key] = selected_players[choice]  # 선택 저장
                return selected_players[choice]
        print("Invalid input. Please try again.")

def Find_ELO(match_idx,team1,team2,file_path="./json/ELO.json"):
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for match_data in data:
        if match_data["Match"] == str(match_idx):
            elo_list = match_data["ELO"]
            team1_elo = None
            team2_elo = None

            for team_data in elo_list:
                if team1 in team_data:
                    team1_elo = team_data[team1]
                if team2 in team_data:
                    team2_elo = team_data[team2]

            if team1_elo is not None and team2_elo is not None:
                return team1_elo, team2_elo
            else:
                print(f"⚠️ 팀 정보를 찾을 수 없음: {team1}, {team2}")
                return None

    print(f"❌ Match ID '{match_idx}'를 찾을 수 없음.")
    return None

def replace_champion_names(pick_list):
    replace_dict = {
    "TwistedFate": "Twisted Fate",
    "XinZhao": "Xin Zhao",
    "RenataGlasc": "Renata Glasc",
    "MissFortune": "Miss Fortune",
    "LeeSin" : "Lee Sin",
    "Dr.Mundo" : "Dr. Mundo",
    "JarvanIV" : "Jarvan IV",
    "AurelionSol": "Aurelion Sol",
    "TahmKench": "Tahm Kench"
    }
    return [replace_dict.get(champ, champ) for champ in pick_list]

def init_weights(region_code, scale=5.0):
    
    region_map = {0: 0, 1: 1, 3: 2}
    if region_code not in region_map:
        raise ValueError(f"Unsupported region_code: {region_code}")
    
    idx = region_map[region_code]
    
    logits = np.zeros(3)
    logits[idx] = scale  # 해당 지역 강조
    
    exp_logits = np.exp(logits)
    weights = exp_logits / np.sum(exp_logits)
    
    return weights
