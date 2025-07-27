import time
import re
import json
import requests
import pandas as pd 
import numpy as np 
import os 

from _utils import * 
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup


chrome_path = r"chromedriver.exe"

def load_match(game_url: str):
    # Selenium 드라이버 설정
    service = Service(chrome_path)
    driver = webdriver.Chrome(service=service)
    driver.get(game_url)
    time.sleep(5)  # JS 로딩 대기

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # ▶ 밴픽 정보 추출
    champion_imgs = soup.select('img[src*="/_img/champions_icon/"]')
    champion_names = [img['src'].split('/')[-1].replace('.png', '') for img in champion_imgs]

    team1_bans = champion_names[0:5]
    team1_picks = champion_names[5:10]
    team2_bans = champion_names[10:15]
    team2_picks = champion_names[15:20]

    # ▶ 골드 차이 추출
    script_tags = soup.find_all("script")
    gold_diff_data = []

    for script in script_tags:
        if "var golddatas" in script.text:
            golddatas_text = re.search(r"var golddatas\s*=\s*(\{.*?\});", script.text, re.DOTALL)
            if golddatas_text:
                js_obj_text = golddatas_text.group(1)
                js_obj_text = re.sub(r"(\w+):", r'"\1":', js_obj_text)  # key에 큰따옴표
                js_obj_text = js_obj_text.replace("'", '"')  # 작은따옴표 -> 큰따옴표
                js_obj_text = re.sub(r",(\s*[}\]])", r"\1", js_obj_text)  # 끝 쉼표 제거

                try:
                    golddatas = json.loads(js_obj_text)
                    for dataset in golddatas.get("datasets", []):
                        if dataset.get("label") == "Gold":
                            gold_diff_data = dataset.get("data", [])
                            break
                except Exception as e:
                    print("JSON parsing error:", e)
            break

    table = soup.find('table', class_='small_table')
    
    if table:
        header_tds = table.find_all('tr')[0].find_all('td')[1:]  # 첫 td는 빈칸
        if len(header_tds) >= 2:
            blue_team_name = header_tds[0].text.strip()
            red_team_name = header_tds[1].text.strip()
        else:
            blue_team_name = red_team_name = None
    else:
        blue_team_name = red_team_name = None
    driver.quit()

    # 결과 반환
    return {
        "blue_bans": team1_bans,
        "blue_picks": team1_picks,
        "red_bans": team2_bans,
        "red_picks": team2_picks,
        "gold_diff": gold_diff_data,
        "B" : blue_team_name,
        "R" : red_team_name
    }

def load_match_saved(path,match_idx,game_name):
    
    match_json_path = path + '/Game/' + match_idx + "/" + game_name
    
    with open(match_json_path, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    return match_data

def load_power(champion_int,lane,region=0,version=123,tier=3):
    # kr : 0 , Eu : 1 , NA : 3
    # 123이 MSI 버젼, 126이 현재
    # 3은 마스터 이상, 2는 에메랄드 이상 
    # 0: top, 1: jg, 2: mid, 3: ad , 4: support 
    url = "https://lol.ps/api/champ/"+str(champion_int)+"/graphs.json"

    params = {
    "region": region,
    "version": version,
    "tier": tier,
    "lane": lane,
    "range": "two_weeks"}

    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://lol.ps/api/champ/"+str(champion_int),
    "Accept": "application/json"}

    response = requests.get(url, params=params, headers=headers)
    
    try:
        data = response.json()
        timeline_winrates = data["data"]["timelineWinrates"]
        time_line_winrates_float = np.array(list(map(float, timeline_winrates)))
        return time_line_winrates_float
    
    except Exception as e:
        print("❌ JSON decode error:", e)
        
def load_rank(champion_name,lane,region=0,path='./data/Rank/'):
    file_path = path + str(region) + '/' + str(lane) + '.csv' 
    df = pd.read_csv(file_path)
    row = df[df['Champion']==champion_name]
    
    if row.empty:
        print(f"❌ '{champion_name}'에 해당하는 랭크 기록이 존재하지 않습니다.")
        return None

    pickban = row['PickBanRate'].values[0]
    winrate = row['WinRate'].values[0]

    return pickban, winrate    

def load_league(champion_name,lane,path='./data/League/'):
    result = {}
    for region in os.listdir(path):
        region_path = os.path.join(path, region)
        file_path = os.path.join(region_path, f"{lane}.csv")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"⚠️ {region} - 파일 읽기 실패: {e}")
            continue
        row = df[df['Champion'] == champion_name]

        if row.empty:
            print(f"❌ '{champion_name}'에 대한 데이터가 {region}에 존재하지 않습니다.")
            result[f'pb_{region}'] = 0
            result[f'wr_{region}'] = -1
        else:
            result[f'pb_{region}'] = row['PickBanRate'].values[0]
            result[f'wr_{region}'] = row['WinRate'].values[0]

    return result

def load_player(champion_name,team,player,path='./data/Gamer/'):
    file_path = path + team + '/' + player + '.csv' 
    df = pd.read_csv(file_path)
    row = df[df['Champion']==champion_name]
    
    if row.empty:
        print(f"❌ '{champion_name}'에 해당하는 게이머 기록이 존재하지 않습니다.")
        return 0,-1

    game = row['Game'].values[0]
    winrate = row['WinRate'].values[0]

    return game, winrate

def load_champion(team,champion,pos_idx):
    
    player_name = Find_player(team,pos_idx)
    champs_idx = Find_champion_idx(champion)
    
    if champs_idx == None:
        print(f"❌ '{champion}'에 해당하는 데이터 인덱스가 존재하지 않습니다.")
    
    game,wr_player = load_player(champion,team,player_name)
    pickban_rank_kr,wr_rank_kr = load_rank(champion,pos_idx,0)
    pickban_rank_eu,wr_rank_eu = load_rank(champion,pos_idx,1)
    pickban_rank_na,wr_rank_na = load_rank(champion,pos_idx,3)
    data_leauge = load_league(champion,pos_idx)
    
    power_graph_kr = load_power(champs_idx,pos_idx,0) / 100.0
    power_graph_eu = load_power(champs_idx,pos_idx,1) / 100.0
    power_graph_na = load_power(champs_idx,pos_idx,3) / 100.0
    
    result = {
        "Gamer": player_name,
        "game_gamer": game,
        "wr_gamer": wr_player,
        "wr_rank_kr": wr_rank_kr,
        "pb_rank_kr": pickban_rank_kr,
        "wr_rank_eu": wr_rank_eu,
        "pb_rank_eu": pickban_rank_eu,
        "wr_rank_na": wr_rank_na,
        "pb_rank_na": pickban_rank_na,
        "po_kr": power_graph_kr,
        "po_eu": power_graph_eu,
        "po_na": power_graph_na,
    }

    result.update(data_leauge)

    return result