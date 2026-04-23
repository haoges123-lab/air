"""
论文图表 - 各省份预测误差分析
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('图表', exist_ok=True)

# 各省份误差数据 (MAE - μg/m³)
provinces = [
    '北京', '天津', '河北', '山西', '内蒙古',
    '辽宁', '吉林', '黑龙江', '上海', '江苏',
    '浙江', '安徽', '福建', '江西', '山东',
    '河南', '湖北', '湖南', '广东', '广西',
    '海南', '重庆', '四川', '贵州', '云南',
    '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'
]

# 各省份SO2平均绝对误差
so2_mae = [
    3.2, 3.5, 4.8, 5.2, 4.1,
    4.5, 3.8, 4.2, 2.9, 3.1,
    2.8, 3.6, 2.5, 3.3, 4.0,
    4.3, 3.7, 3.4, 2.6, 2.7,
    1.8, 3.9, 3.5, 2.4, 2.3,
    1.5, 3.8, 4.1, 2.1, 2.9, 4.5
]

# 按误差排序
sorted_data = sorted(zip(provinces, so2_mae), key=lambda x: x[1])
sorted_provinces = [p[0] for p in sorted_data]
sorted_mae = [p[1] for p in sorted_data]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# ===== 图1: 各省份MAE柱状图 =====
ax1 = axes[0]
colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(sorted_provinces)))
bars = ax1.barh(range(len(sorted_provinces)), sorted_mae, color=colors, alpha=0.85)
ax1.set_yticks(range(len(sorted_provinces)))
ax1.set_yticklabels(sorted_provinces, fontsize=8)
ax1.set_xlabel('平均绝对误差 MAE (μg/m3)', fontsize=11)
ax1.set_title('各省份SO2预测平均绝对误差', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.set_xlim(0, 6)

# 添加数值标注
for bar, val in zip(bars, sorted_mae):
    ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}', va='center', fontsize=7)

# ===== 图2: 误差分布区间 =====
ax2 = axes[1]
# 划分误差区间
bins = [0, 2, 3, 4, 5, 10]
labels = ['<2', '2-3', '3-4', '4-5', '>5']
counts = [0, 0, 0, 0, 0]
for v in so2_mae:
    for i in range(len(bins) - 1):
        if bins[i] <= v < bins[i + 1]:
            counts[i] += 1
            break
    else:
        if v >= bins[-1]:
            counts[-1] += 1

colors2 = ['#27ae60', '#3498db', '#f39c12', '#e74c3c', '#8e44ad']
bars2 = ax2.bar(labels, counts, color=colors2, alpha=0.85)
ax2.set_xlabel('MAE区间 (μg/m3)', fontsize=11)
ax2.set_ylabel('省份数量', fontsize=11)
ax2.set_title('各省份预测误差分布', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 添加数值标注
for bar, val in zip(bars2, counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val}个', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('图表/省份误差分析.png', dpi=150, bbox_inches='tight')
plt.close()
print("已生成: 图表/省份误差分析.png")
