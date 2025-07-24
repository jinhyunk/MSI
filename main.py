import argparse
from _utils import * 

def main(args):
    args = args
    #device = args.device
    path_data = args.data
    match_idx = args.idx
    

    match_json_path = path_data + '/Game/' + match_idx + "/1_B_GEN_R_G2"
    
    with open(match_json_path, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    print(match_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data', default='./data', type=str, help='Data location')
    parser.add_argument('--idx', default='1', type=str, help='match idx (1~14 , pi_1~pi_5)')
    args = parser.parse_args()
    main(args)