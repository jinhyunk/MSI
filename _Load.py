# data_loader_refactored.py
import os
import json
import re
import time
import requests
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from functools import lru_cache

# --- 상수 정의 ---
PB_NEW = 0.005
WR_NEW = 0.50

# --- 헬퍼 함수 (이전 data_loader.py에서 가져옴) ---
from _utils import find_player, find_champion_id, DataManager as BaseDataManager

# ==============================================================================
# 1. 데이터 캐싱 및 관리 클래스
# ==============================================================================
class DataManager(BaseDataManager):
    """
    기존 DataManager를 확장하여 Rank, League, Gamer CSV 데이터도 캐싱합니다.
    """
    def __init__(self, base_path="./data/"):
        # BaseDataManager의 __init__ 호출 (json 로드)
        super().__init__(base_path=os.path.join(base_path, 'json'))
        self.data_path = base_path
        
        # CSV 데이터 캐싱
        self._league_data = self._load_csv_data(os.path.join(base_path, 'League'))
        self._rank_data = self._load_csv_data(os.path.join(base_path, 'Rank'))
        self._gamer_data = self._load_gamer_data(os.path.join(base_path, 'Gamer'))

    def _load_csv_data(self, path):
        """리그, 랭크 데이터를 딕셔너리 형태로 미리 로드합니다."""
        data_cache = {}
        if not os.path.exists(path): return data_cache
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            data_cache[folder] = {}
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    lane = file.replace('.csv', '')
                    file_path = os.path.join(folder_path, file)
                    data_cache[folder][lane] = pd.read_csv(file_path).set_index('Champion')
        return data_cache

    def _load_gamer_data(self, path):
        """게이머 데이터를 미리 로드합니다."""
        data_cache = {}
        if not os.path.exists(path): return data_cache
        for team_folder in os.listdir(path):
            team_path = os.path.join(path, team_folder)
            data_cache[team_folder] = {}
            for file in os.listdir(team_path):
                if file.endswith('.csv'):
                    player = file.replace('.csv', '')
                    file_path = os.path.join(team_path, file)
                    data_cache[team_folder][player] = pd.read_csv(file_path).set_index('Champion')
        return data_cache

    def get_league_stats(self, league, lane, champion):
        df = self._league_data.get(league, {}).get(str(lane))
        if df is None or champion not in df.index:
            return {'pb': PB_NEW, 'wr': WR_NEW}
        row = df.loc[champion]
        return {'pb': row['PickBanRate'], 'wr': row['WinRate']}

    def get_rank_stats(self, region_code, lane, champion):
        df = self._rank_data.get(str(region_code), {}).get(str(lane))
        if df is None or champion not in df.index:
            return {'pb': PB_NEW, 'wr': WR_NEW}
        row = df.loc[champion]
        return {'pb': row['PickBanRate'], 'wr': row['WinRate']}

    def get_mastery_stats(self, team, player, champion):
        df = self._gamer_data.get(team, {}).get(player)
        if df is None or champion not in df.index:
            return {'game': PB_NEW, 'wr': WR_NEW}
        row = df.loc[champion]
        return {'game': row['Game'], 'wr': row['WinRate']}

# ==============================================================================
# 2. PO 데이터 관리 클래스 (API, 파일 I/O, 계산)
# ==============================================================================
class PowerManager:
    def __init__(self, data_manager, po_json_path="./json/po/"):
        self.data_manager = data_manager
        self.po_path = po_json_path
        os.makedirs(self.po_path, exist_ok=True)

    def get_po(self, time_val, champion, lane, version=123, tier=3, s=0):
        """PO 데이터를 가져오거나, 없으면 API 호출 후 생성하여 반환합니다."""
        po_data = self.find_po_from_file(champion, lane)
        
        if po_data is None:
            print(f"ℹ️ 로컬에서 '{champion}' PO 데이터를 찾을 수 없어 API를 호출합니다.")
            champion_id = find_champion_id(self.data_manager, champion)
            if champion_id is None:
                print(f"❌ '{champion}'의 ID를 찾을 수 없습니다.")
                return None # 혹은 기본값
            
            api_data = self._fetch_from_api(champion_id, lane, version, tier)
            po_data = self._save_po_to_file(champion, lane, api_data, s)
            if po_data is None:
                return None

        return self._calculate_po_at_time(time_val, po_data)

    def find_po_from_file(self, champion, lane):
        file_path = os.path.join(self.po_path, f"{lane}.json")
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for champ_name, champ_data in data.items():
            if champ_name.lower() == champion.lower():
                return champ_data
        return None
        
    def _fetch_from_api(self, champion_id, lane, version, tier):
        """lol.ps API에서 데이터를 가져옵니다."""
        result = {'time': np.arange(5.0, 36.0, 1.0) / 35.0}
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        
        for region_code, region_name in self.data_manager.REGION_MAP.items():
            url = f"https://lol.ps/api/champ/{champion_id}/graphs.json"
            params = {"region": region_code, "version": version, "tier": tier, "lane": lane, "range": "two_weeks"}
            try:
                response = requests.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()["data"]["timelineWinrates"]
                result[f"po_rank_{region_name}"] = np.array(list(map(float, data))) / 100.0
            except Exception as e:
                print(f"❌ {region_name} - API Error: {e}")
                result[f"po_rank_{region_name}"] = None
        return result

    def _save_po_to_file(self, champion, lane, api_data, s):
        """API 데이터를 JSON 파일에 저장합니다."""
        file_path = os.path.join(self.po_path, f"{lane}.json")
        lane_data = {}
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try: lane_data = json.load(f)
                except json.JSONDecodeError: pass

        lane_data[champion] = {}
        for region in self.data_manager.REGION_MAP.values():
            y_data = api_data.get(f"po_rank_{region}")
            if y_data is not None:
                lane_data[champion][region] = {
                    "time": api_data["time"].tolist(),
                    "y": y_data.tolist(),
                    "s": s
                }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(lane_data, f, ensure_ascii=False, indent=2)
        print(f"✅ '{champion}'의 PO 데이터를 '{file_path}'에 저장했습니다.")
        return lane_data[champion]

    def _calculate_po_at_time(self, time_val, po_data):
        """주어진 시간 값에 대한 PO를 계산합니다."""
        if hasattr(time_val, 'cpu'): time_val = time_val.cpu().numpy()
        time_val = float(time_val)
        
        output = {}
        for region, data in po_data.items():
            spline = UnivariateSpline(data["time"], data["y"], s=data["s"])
            output[f'po_{region}'] = float(spline(time_val))
        return output

# ==============================================================================
# 3. 데이터 로더 함수 (리팩토링된 버전)
# ==============================================================================
def load_player_data(data_manager, team, champion, pos_idx):
    """
    DataManager를 사용하여 플레이어의 숙련도 정보를 로드합니다.
    (기존 load_player, load_mastery 통합)
    """
    player_name = find_player(data_manager, team, pos_idx)
    if not player_name:
        return {"Gamer": None, "game_gamer": PB_NEW, "wr_gamer": WR_NEW}
        
    mastery = data_manager.get_mastery_stats(team, player_name, champion)
    return {
        "Gamer": player_name,
        "game_gamer": mastery['game'],
        "wr_gamer": mastery['wr'],
    }

def _load_match_data(path):
    """(내부 함수) Train/Test 매치 데이터 로딩 로직 통합"""
    result = []
    if not os.path.exists(path): return result
    for match_dir in os.listdir(path):
        match_path = os.path.join(path, match_dir)
        game_files = [f for f in os.listdir(match_path) if not f.startswith('.')]
        data_game = []
        for file in game_files:
            file_path = os.path.join(match_path, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    game_data = json.load(f)
                data_game.append({"game_name": file, "game_data": game_data})
            except Exception as e:
                print(f"❌ '{file_path}' 로드 실패: {e}")
        result.append({"match_idx": match_dir, "data_game": data_game})
    return result

def load_match_train(path="./data/Game/"):
    return _load_match_data(path)

def load_match_test(path="./data/Game_test_final/"):
    return _load_match_data(path)

# ==============================================================================
# 4. 웹 스크래핑 모듈 (독립적으로 분리)
# ==============================================================================
class WebScraper:
    def __init__(self, driver_path="chromedriver.exe"):
        self.driver_path = driver_path

    def scrape_gol_gg(self, game_url):
        """gol.gg에서 게임 데이터를 스크래핑합니다."""
        service = Service(self.driver_path)
        driver = webdriver.Chrome(service=service)
        driver.get(game_url)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()
        
        # ... (기존 load_match_golgg의 파싱 로직과 동일)
        # ▶ 밴픽 정보 추출 ...
        # ▶ 골드 차이 추출 ...
        # ▶ 팀 이름 추출 ...
        # 이 부분은 길어서 생략했지만, 원래 코드를 여기에 그대로 붙여넣으면 됩니다.
        
        # 임시 반환 값
        return {"message": "Scraping logic should be placed here."}