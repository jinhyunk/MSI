import requests

champion_int = 902

url = "https://lol.ps/api/champ/"+str(champion_int)+"/graphs.json"

region = 0   # kr : 0 , Eu : 1 , NA : 2
version = 123 # 123이 MSI 버젼, 126이 현재
tier = 3 # 3은 마스터 이상, 2는 에메랄드 이상 
lane = 4 # 0: top, 1: jg, 2: mid, 3: ad , 4: support 

params = {
    "region": region,
    "version": version,
    "tier": tier,
    "lane": 4,
    "range": "two_weeks"
}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://lol.ps/champ/2",
    "Accept": "application/json"
}

response = requests.get(url, params=params, headers=headers)

# 디버깅 출력
print("Status Code:", response.status_code)
print("Response Sample:", response.text[:200])

# JSON 처리
try:
    data = response.json()
    timeline_winrates = data["data"]["timelineWinrates"]

    print("\n🕒 Timeline Winrates:")
    for i, wr in enumerate(timeline_winrates):
        minute = 5 + i 
        print(f"{minute}분: {wr}%")
except Exception as e:
    print("❌ JSON decode error:", e)
