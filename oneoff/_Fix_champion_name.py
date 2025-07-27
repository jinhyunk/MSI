import os
import json

def replace_names_in_list(lst, replace_dict):
    return [replace_dict.get(x, x) for x in lst]

def process_json_file(file_path, replace_dict):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # JSON이 아니거나 읽기 실패 시 무시
        return

    changed = False
    keys_to_check = ["blue_bans", "blue_picks", "red_bans", "red_picks"]

    for key in keys_to_check:
        if key in data and isinstance(data[key], list):
            new_list = replace_names_in_list(data[key], replace_dict)
            if new_list != data[key]:
                data[key] = new_list
                changed = True

    if changed:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Updated {file_path}")

def process_all_json_like_files(root_folder="./data/Game", replace_dict=None):
    if replace_dict is None:
        replace_dict = {}
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            # 확장자 검사 없이 모든 파일 시도
            process_json_file(full_path, replace_dict)

replace_names = {
    "TwistedFate": "Twisted Fate",
    "XinZhao": "Xin Zhao",
    "RenataGlasc": "Renata Glasc",
    "MissFortune": "Miss Fortune",
    "LeeSin" : "Lee Sin",
    "Dr.Mundo" : "Dr. Mundo",
    "JarvanIV" : "Jarvan IV",
    "AurelionSol": "Aurelion Sol"
}

process_all_json_like_files(replace_dict=replace_names)
