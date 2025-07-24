from _utils import * 

def load_match(path,match_idx,game_name):
    
    match_json_path = path + '/Game/' + match_idx + "/" + game_name
    
    with open(match_json_path, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    return match_data

