import time
import re
import json
import requests
import pandas as pd 
import numpy as np 
import os 

from _utils import * 
from _Config import leagues,regions,region_map
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from scipy.interpolate import UnivariateSpline

chrome_path = r"chromedriver.exe"
region_map = {'0': 'kr', '1': 'eu', '3': 'na'}
pb_new = 0.005
wr_new = 0.50


def load_match_golgg(game_url: str):
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

def load_match_train(path="./data/Game/"):
    result = []
    for match in os.listdir(path):
        match_path = os.path.join(path, match)
        data_game = []
        for file in os.listdir(match_path):
            file_path = os.path.join(match_path, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    game_data = json.load(f)
                data_game.append({
                    "game_name": file,
                    "game_data": game_data
                })
            except Exception as e:
                print(f"❌ Failed to load '{file_path}':", e)
        match_data = {
            "match_idx": match,
            "data_game": data_game
        }
        result.append(match_data)
    return result

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
            result[f'pb_lg_{region}'] = pb_new
            result[f'wr_lg_{region}'] = wr_new
        else:
            result[f'pb_lg_{region}'] = row['PickBanRate'].values[0]
            result[f'wr_lg_{region}'] = row['WinRate'].values[0]

    return result

def load_rank(champion_name, lane, path='./data/Rank/'):
    result = {}
    
    for region_code in os.listdir(path):
        if region_code not in region_map:
            print(f"⚠️ Unknown region code: {region_code}, skipping.")
            continue

        region_name = region_map[region_code]
        region_path = os.path.join(path, region_code)
        file_path = os.path.join(region_path, f"{lane}.csv")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"⚠️ {region_name} - 파일 읽기 실패: {e}")
            continue

        row = df[df['Champion'] == champion_name]

        if row.empty:
            result[f'pb_rank_{region_name}'] = pb_new
            result[f'wr_rank_{region_name}'] = wr_new
        else:
            result[f'pb_rank_{region_name}'] = row['PickBanRate'].values[0]
            result[f'wr_rank_{region_name}'] = row['WinRate'].values[0]

    return result

def load_mastery(champion_name,team,player,path='./data/Gamer/'):
    file_path = path + team + '/' + player + '.csv' 
    df = pd.read_csv(file_path)
    row = df[df['Champion']==champion_name]
    
    if row.empty:
        return pb_new, wr_new

    game = row['Game'].values[0]
    winrate = row['WinRate'].values[0]

    return game, winrate

def load_player(team,champion,pos_idx):
    player_name = Find_player(team,pos_idx)
    champs_idx = Find_champion_idx(champion)
    
    if champs_idx == None:
        print(f"❌ '{champion}'에 해당하는 데이터 인덱스가 존재하지 않습니다.")
    game,wr_player = load_mastery(champion,team,player_name)
    
    return {
        "Gamer": player_name,
        "game_gamer": game,
        "wr_gamer": wr_player,
    }

def load_power_ps(champion_int,lane,version=123,tier=3):
    result = {}
    time_key = f'time'
    times = np.arange(5.0, 36.0, 1.0) / 35.0
    result[time_key] = times

    for region_code, region_name in region_map.items():
        url = f"https://lol.ps/api/champ/{champion_int}/graphs.json"
        params = {
            "region": region_code,
            "version": version,
            "tier": tier,
            "lane": lane,
            "range": "two_weeks"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": f"https://lol.ps/api/champ/{champion_int}",
            "Accept": "application/json"
        }

        try:
            response = requests.get(url, params=params, headers=headers)
            data = response.json()
            timeline_winrates = data["data"]["timelineWinrates"]
            timeline_winrates_float = np.array(list(map(float, timeline_winrates)))
            result[f"po_rank_{region_name}"] = timeline_winrates_float / 100.0
        except Exception as e:
            print(f"❌ {region_name} - JSON decode error or missing data:", e)
            result[f"po_rank_{region_name}"] = None  # 또는 np.zeros(n) 등 기본값으로 설정 가능

    return result

def save_power(champion_name, lane_num, data, s=0, save_dir="./json/po/"):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{lane_num}.json")

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lane_data = json.load(f)
    else:
        lane_data = {}

    lane_data[champion_name] = {}
    for region in region_map.values():
        y = data[f"po_rank_{region}"]
        if y is not None:
            spline = UnivariateSpline(data["time"], y, s=s)
            # spline은 저장할 때 x, y 원본을 저장해서 나중에 재생성
            lane_data[champion_name][region] = {
                "time": data["time"].tolist(),
                "y": y.tolist(),
                "s": s
            }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(lane_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved spline for {champion_name} in {lane_num}")
    
    return lane_data[champion_name]

def calc_po(time_value,po_data):
    output = {}
    # GPU 텐서인 경우 CPU로 이동 후 numpy로 변환
    if hasattr(time_value, 'device') and time_value.device.type == 'cuda':
        time_value = time_value.cpu().numpy()
    elif hasattr(time_value, 'numpy'):
        time_value = time_value.numpy()
    else:
        time_value = float(time_value)
    
    for region in regions:
        po_key = f'{region}'
        out_key = f'po_{region}'
        reg_data = po_data[po_key]
        spline = UnivariateSpline(reg_data["time"], reg_data["y"], s=reg_data["s"])
        output[out_key] = float(spline(time_value))
    
    return output

def load_power(time,champion,lane,version=123,tier=3,s=0):
    data = Find_po(champion,lane)
    if data is None:
        champion_idx = Find_champion_idx(champion)

        if champion_idx is None:
            print(f"Champion {champion} not found in champion json.")
            exit()
    
        print(f"Champion {champion} po is not found in po json.")
        data_ps = load_power_ps(champion_idx,lane,version,tier)
        data = save_power(champion,lane,data_ps,s)
    
    po_time = calc_po(time,data)

    return po_time 
