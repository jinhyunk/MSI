import requests

def load_power(champion_int,lane,region=0,version=123,tier=3):
    
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
        return timeline_winrates
    
    except Exception as e:
        print("❌ JSON decode error:", e)
