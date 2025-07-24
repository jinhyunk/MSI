from utils import *
import os 

url = "https://gol.gg/game/stats/69327/page-game/"

data = Dataloader(url)

save_path = './csv_data/Game/test/4_B_T1_R_GEN'

try:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 폴더 없으면 생성
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"데이터가 저장되었습니다: {save_path}")
    
except Exception as e:
    print(f"저장 실패: {e}")
    

