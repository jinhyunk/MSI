from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import re
import json
import math

chrome_path = r"chromedriver.exe"

def Dataloader(game_url: str):
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

    driver.quit()

    # 결과 반환
    return {
        "blue_bans": team1_bans,
        "blue_picks": team1_picks,
        "red_bans": team2_bans,
        "red_picks": team2_picks,
        "gold_diff": gold_diff_data
    }


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


def Find_champion_idx(name_us, file_path="champions.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for champ in data:
        if champ["nameUs"].lower() == name_us.lower():
            return champ["championId"]
    
    return None  # 찾지 못했을 경우
