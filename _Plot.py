import matplotlib.pyplot as plt

def graph_team_time_wr(time_wr,threshold=0.50):
    x = list(range(5, 5 + len(time_wr)))
    
    plt.figure(figsize=(10,5))
    plt.axhline(threshold, color='gray', linestyle='--')

    # 전체 선 그리기 (하나의 연속된 선)
    plt.plot(x, time_wr, linestyle='-', color='black', alpha=0.3, label='Total Power Line')

    # 점들은 조건에 따라 색상 다르게 표시
    colors = ['blue' if val > threshold else 'red' for val in time_wr]
    plt.scatter(x, time_wr, color=colors, label='Power Points')

    plt.xlabel('Time')
    plt.ylabel('Total Power')
    plt.title('Total Power Over Time with Threshold Coloring')
    plt.legend()
    plt.grid(True)
    plt.show()

def graph_compare_time_wr(blue_wr,red_wr):
    x = list(range(5, 5 + len(blue_wr)))
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

def graph_compare_time_gold(gold_diff):
    x = list(range(len(gold_diff)))

    plt.plot(x, gold_diff, color='gray', linestyle='-', linewidth=1)

    # 점 찍기 (색 구분)
    pos_x = [i for i, v in enumerate(gold_diff) if v > 0]
    pos_y = [gold_diff[i] for i in pos_x]

    neg_x = [i for i, v in enumerate(gold_diff) if v < 0]
    neg_y = [gold_diff[i] for i in neg_x]

    zero_x = [i for i, v in enumerate(gold_diff) if v == 0]
    zero_y = [0]*len(zero_x)

    plt.scatter(pos_x, pos_y, color='blue', label='Positive (>0)')
    plt.scatter(neg_x, neg_y, color='red', label='Negative (<0)')
    plt.scatter(zero_x, zero_y, color='gray', label='Zero (=0)')

    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Gold Difference')
    plt.title('Gold Difference Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()