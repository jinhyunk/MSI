import json
import math
import pandas as pd 


def calc_winrate(elo_team1, elo_team2):
    """
    두 팀의 ELO 레이팅을 기반으로 팀1이 승리할 확률을 계산합니다.
    
    Parameters:
    - elo_team1 (float): 팀1의 ELO 레이팅
    - elo_team2 (float): 팀2의 ELO 레이팅

    Returns:
    - float: 팀1이 승리할 확률 (0 ~ 1)
    """
    probability = 1 / (1 + math.pow(10, (elo_team2 - elo_team1) / 400))
    return probability

def Find_champion_idx(name_us, file_path="./json/champions.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for champ in data:
        if champ["nameUs"].lower() == name_us.lower():
            return champ["championId"]
    
    return None  # 찾지 못했을 경우

def Find_player(team_name, pos_idx, file_path="./json/roster.json"):
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
