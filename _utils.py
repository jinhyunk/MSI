# data_loader.py
import os
import json
import torch
from functools import lru_cache

# --- 유틸리티성 딕셔너리 및 함수 ---

POSITION_NAMES = {0: "Top", 1: "Jungle", 2: "Mid", 3: "ADC", 4: "Support"}

REPLACE_CHAMPION_NAMES_DICT = {
    "TwistedFate": "Twisted Fate", "XinZhao": "Xin Zhao", "RenataGlasc": "Renata Glasc",
    "MissFortune": "Miss Fortune", "LeeSin": "Lee Sin", "Dr.Mundo": "Dr. Mundo",
    "JarvanIV": "Jarvan IV", "AurelionSol": "Aurelion Sol", "TahmKench": "Tahm Kench",
    "KSante": "K'Sante", "Kaisa": "Kai'Sa", "Chogath": "Cho'Gath",
    "KhaZix": "Kha'Zix", "Nunu": "Nunu & Willump"
}

def replace_champion_names(pick_list):
    """챔피언 이름을 표준 이름으로 교체합니다."""
    return [REPLACE_CHAMPION_NAMES_DICT.get(champ, champ) for champ in pick_list]

# --- 데이터 관리 클래스 ---

class DataManager:
    """
    프로젝트에서 사용하는 모든 JSON 데이터를 로드하고 캐싱하는 중앙 관리 클래스.
    """
    def __init__(self, base_path="./json/"):
        self.base_path = base_path
        self._champions = self._load_json("champions.json")
        self._roster = self._load_json("roster.json")
        self._elo = self._load_json("ELO.json")
        self.player_selection_cache = {}

    def _load_json(self, file_name):
        """JSON 파일을 안전하게 로드합니다."""
        path = os.path.join(self.base_path, file_name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: '{path}' 파일을 찾을 수 없습니다.")
            return [] if isinstance(file_name, str) and 's' in file_name else {}
        except json.JSONDecodeError:
            print(f"❌ Error: '{path}' 파일의 형식이 잘못되었습니다.")
            return [] if isinstance(file_name, str) and 's' in file_name else {}

    @lru_cache(maxsize=128) # 같은 챔피언, 포지션 검색 시 캐시된 결과 바로 반환
    def find_po_data(self, champion, pos_idx):
        """특정 포지션의 챔피언 파워(PO) 데이터를 로드합니다."""
        po_path = os.path.join(self.base_path, "po", f"{pos_idx}.json")
        if not os.path.exists(po_path):
            return None
        
        with open(po_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # 대소문자 구분 없이 챔피언 검색
        for champ_name, champ_data in data.items():
            if champ_name.lower() == champion.lower():
                return champ_data
        return None

    def get_champions(self):
        return self._champions

    def get_roster(self):
        return self._roster
        
    def get_elo(self):
        return self._elo

# --- 데이터 조회 함수 ---

def find_champion_id(data_manager, champion_name_us):
    """챔피언 영문 이름으로 고유 ID를 찾습니다."""
    for champ in data_manager.get_champions():
        if champ["nameUs"].lower() == champion_name_us.lower():
            return champ["championId"]
    return None

def find_player(data_manager, team_name, pos_idx, auto_select_index=0):
    """
    팀과 포지션에 맞는 선수를 찾습니다. 
    여러 명일 경우, 캐시된 선택이나 자동 선택(기본 첫 번째)에 따라 반환합니다.
    """
    key = (team_name, pos_idx)
    if key in data_manager.player_selection_cache:
        return data_manager.player_selection_cache[key]

    roster = data_manager.get_roster()
    selected_players = []

    for team in roster:
        if team["team"] == team_name:
            selected_players = [p["name"] for p in team.get("players", []) if p["position"] == pos_idx]
            break
            
    if not selected_players:
        pos_name = POSITION_NAMES.get(pos_idx, 'Unknown')
        print(f"⚠️ '{team_name}' 팀의 '{pos_name}' 포지션 선수를 찾을 수 없습니다.")
        return None

    if len(selected_players) == 1:
        player = selected_players[0]
    else:
        # 여러 플레이어가 있을 경우, 대화형 입력 대신 첫 번째 플레이어를 기본으로 선택
        print(f"ℹ️ '{team_name}' 팀 '{POSITION_NAMES.get(pos_idx)}' 포지션에 여러 선수가 등록되어 자동 선택합니다: {selected_players}")
        player = selected_players[auto_select_index]
        
    data_manager.player_selection_cache[key] = player
    return player

def find_elo(data_manager, match_idx, team_name):
    """매치 ID와 팀 이름으로 ELO 점수를 찾습니다."""
    elo_data = data_manager.get_elo()
    match_id_str = str(match_idx)

    for match_data in elo_data:
        if match_data.get("Match") == match_id_str:
            for team_elo_map in match_data.get("ELO", []):
                if team_name in team_elo_map:
                    elo = team_elo_map[team_name]
                    return torch.tensor([elo], dtype=torch.float32).view(-1, 1)
            print(f"⚠️ '{match_id_str}' 매치에서 '{team_name}' 팀의 ELO를 찾을 수 없습니다.")
            return None
            
    print(f"❌ Match ID '{match_id_str}'를 찾을 수 없습니다.")
    return None