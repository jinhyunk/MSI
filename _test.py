import matplotlib.pyplot as plt
import numpy as np 

from _utils import * 
from _Load_match import * 
from _Load_lolps import * 

data_path = "./data"
match_idx = "1"
game_name = "1_B_GEN_R_G2"

data = load_match(data_path,match_idx,game_name)

checker = "blue_picks"
#checker = "red_picks"

pick_data = data[checker]
total_power = None

for i in range(0,len(pick_data)):
    print(i)
    champs_name = pick_data[i]
    champs_idx = Find_champion_idx(champs_name)
    if champs_idx == None:
        print("Error w. champs_idx")

    power_graph = load_power(champs_idx,i)
    power_float = np.array(list(map(float, power_graph))) 
    if total_power is None:
        total_power = np.zeros_like(power_float) 

    total_power = total_power + power_float
    
total_power = total_power/ 5.0

# 이후 플롯 코드 그대로
x = list(range(5, 5 + len(total_power)))
threshold = 50

plt.figure(figsize=(10,5))
plt.axhline(threshold, color='gray', linestyle='--')

# 전체 선 그리기 (하나의 연속된 선)
plt.plot(x, total_power, linestyle='-', color='black', alpha=0.3, label='Total Power Line')

# 점들은 조건에 따라 색상 다르게 표시
colors = ['blue' if val > threshold else 'red' for val in total_power]
plt.scatter(x, total_power, color=colors, label='Power Points')

plt.xlabel('Time')
plt.ylabel('Total Power')
plt.title('Total Power Over Time with Threshold Coloring')
plt.legend()
plt.grid(True)
plt.show()