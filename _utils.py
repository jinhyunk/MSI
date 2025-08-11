import json
import math
import torch 

import pandas as pd 
import numpy as np 
import os 

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

def Find_ELO(match_idx,team,file_path="./json/ELO.json"):
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for match_data in data:
        if match_data["Match"] == str(match_idx):
            elo_list = match_data["ELO"]
            team_elo = None

            for team_data in elo_list:
                if team in team_data:
                    team_elo = team_data[team]

            if team_elo is not None:
                ELO = torch.tensor([team_elo], dtype=torch.float32)
                ELO = ELO.view(-1, 1)
                return ELO
            else:
                print(f"⚠️ 팀 정보를 찾을 수 없음: {team}")
                return None

    print(f"❌ Match ID '{match_idx}'를 찾을 수 없음.")
    return None

def Find_po(champion,pos_idx,file_path="./json/po/"):
    file_path_line = file_path + f"{pos_idx}.json"
    
    if not os.path.exists(file_path_line):
        return None
    
    with open(file_path_line, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for champ_name in data.keys():
        if champ_name.lower() == champion.lower():
            return data[champ_name]  # 챔피언 데이터 전체 반환
    
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
    "TahmKench": "Tahm Kench",
    "KSante": "K'Sante",
    "Kaisa": "Kai'Sa",
    "Chogath":"Cho'Gath",
    "KhaZix" : "Kha'Zix",
    "Nunu" : "Nunu & Willump"
    }
    return [replace_dict.get(champ, champ) for champ in pick_list]

