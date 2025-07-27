import argparse
from _Calc import * 
from _Load import * 
from _utils import * 
from _Plot import * 

def main(args):
    args = args
    #device = args.device
    path_data = args.data
    idx_match = args.idx
    
    blue_ELO = 1807  ## 이거는 자동화 가능하려나..?
    red_ELO = 1744

    wr_B, wr_R = calc_winrate(blue_ELO,red_ELO)
    print(wr_B)
    print(wr_R)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device', default='cuda', type=str, help='cuda / cpu')
    parser.add_argument('--data', default='./data', type=str, help='Data location')
    parser.add_argument('--idx', default='69297', type=str, help='gol.gg match idx 69322/69314/69297')
    args = parser.parse_args()
    main(args)