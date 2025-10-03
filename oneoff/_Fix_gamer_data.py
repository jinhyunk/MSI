import os
import pandas as pd

root_dir = "./data/Gamer"

for team_name in os.listdir(root_dir):
    team_path = os.path.join(root_dir, team_name)
    if not os.path.isdir(team_path):
        continue  # 팀 폴더가 아닐 경우 무시

    for file_name in os.listdir(team_path):
        if not file_name.endswith(".csv"):
            continue  # csv 파일만 처리

        file_path = os.path.join(team_path, file_name)
        try:
            df = pd.read_csv(file_path)

            # 헤더에 'Win rate'가 있을 경우만 수정
            if "Win rate" in df.columns:
                df.rename(columns={"Win rate": "WinRate"}, inplace=True)
                df.to_csv(file_path, index=False)
                print(f"✅ Updated: {file_path}")
            else:
                print(f"ℹ️ Skipped (no 'Win rate'): {file_path}")
        except Exception as e:
            print(f"❌ Error in {file_path}: {e}")
