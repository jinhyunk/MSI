import matplotlib.pyplot as plt

from _Calc_time import *


data_path = "./data"
match_idx = "1"
game_name = "1_B_GEN_R_G2"

blue_wr, red_wr = Calc_time_pick(data_path,match_idx,game_name)

x = list(range(5, 5 + len(blue_wr)))
threshold = 50

plt.figure(figsize=(10,5))
plt.axhline(threshold, color='gray', linestyle='--')

# 전체 선 그리기 (하나의 연속된 선)
plt.plot(x, blue_wr, linestyle='-', color='black', alpha=0.3, label='Total Power Line')

# 점들은 조건에 따라 색상 다르게 표시
colors = ['blue' if val > threshold else 'red' for val in blue_wr]
plt.scatter(x, blue_wr, color=colors, label='Power Points')

plt.xlabel('Time')
plt.ylabel('Total Power')
plt.title('Total Power Over Time with Threshold Coloring')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.axhline(threshold, color='gray', linestyle='--')

# 전체 선 그리기 (하나의 연속된 선)
plt.plot(x, red_wr, linestyle='-', color='black', alpha=0.3, label='Total Power Line')

# 점들은 조건에 따라 색상 다르게 표시
colors = ['blue' if val > threshold else 'red' for val in red_wr]
plt.scatter(x, red_wr, color=colors, label='Power Points')

plt.xlabel('Time')
plt.ylabel('Total Power')
plt.title('Total Power Over Time with Threshold Coloring')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))

# blue 팀 선 그래프 (파란색)
plt.plot(x, blue_wr, linestyle='-', color='blue', label='Blue Winrate')
plt.scatter(x, blue_wr, color='blue')

# red 팀 선 그래프 (빨간색)
plt.plot(x, red_wr, linestyle='-', color='red', label='Red Winrate')
plt.scatter(x, red_wr, color='red')

plt.xlabel('Time')
plt.ylabel('Winrate')
plt.title('Blue vs Red Winrate Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()