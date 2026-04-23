"""
系统功能模块与流程图生成
空气质量预测系统 - 架构图和流程图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import matplotlib.lines as mlines
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ============ 图1: 系统功能模块图 ============
fig1, ax1 = plt.subplots(1, 1, figsize=(18, 12))
ax1.set_xlim(0, 18)
ax1.set_ylim(0, 12)
ax1.axis('off')
ax1.set_title('空气质量预测系统 - 功能模块架构图', fontsize=20, fontweight='bold', pad=20, color='#333')

# 颜色定义
colors = {
    'frontend': '#4ECDC4',      # 青色 - 前端
    'backend': '#667EEA',       # 紫色 - 后端
    'database': '#45B7D1',      # 蓝色 - 数据库
    'api': '#FF6B6B',           # 红色 - 外部API
    'ai': '#FFD93D',            # 黄色 - AI服务
    'cache': '#A29BFE',         # 浅紫 - 缓存
}

# --- 前端模块 (左侧) ---
frontend_box = FancyBboxPatch((0.5, 3), 3.5, 7,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    facecolor=colors['frontend'], alpha=0.3, edgecolor=colors['frontend'], linewidth=2)
ax1.add_patch(frontend_box)
ax1.text(2.25, 9.5, '前端展示层', fontsize=14, fontweight='bold', ha='center', color=colors['frontend'])

# 前端子模块
frontend_modules = [
    ('地图可视化', 8.5),
    ('城市列表', 7),
    ('预测界面', 5.5),
    ('数据分析', 4),
]
for name, y in frontend_modules:
    box = FancyBboxPatch((0.8, y), 2.9, 0.8,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor='white', edgecolor=colors['frontend'], linewidth=1.5)
    ax1.add_patch(box)
    ax1.text(2.25, y+0.4, name, fontsize=11, ha='center', va='center', color='#333')

# --- 中央后端模块 (中间) ---
backend_box = FancyBboxPatch((6.5, 2.5), 4, 8,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    facecolor=colors['backend'], alpha=0.3, edgecolor=colors['backend'], linewidth=2)
ax1.add_patch(backend_box)
ax1.text(8.5, 10, '后端服务层', fontsize=14, fontweight='bold', ha='center', color=colors['backend'])

# 后端子模块
backend_modules = [
    ('FastAPI API网关', 8.5),
    ('数据处理模块', 7),
    ('预测引擎', 5.5),
    ('建议生成器', 4),
    ('缓存管理', 3),
]
for name, y in backend_modules:
    box = FancyBboxPatch((6.8, y), 3.4, 0.7,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor='white', edgecolor=colors['backend'], linewidth=1.5)
    ax1.add_patch(box)
    ax1.text(8.5, y+0.35, name, fontsize=11, ha='center', va='center', color='#333')

# --- 数据库与缓存 (右侧) ---
db_box = FancyBboxPatch((12.5, 6), 2.5, 3.5,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    facecolor=colors['database'], alpha=0.3, edgecolor=colors['database'], linewidth=2)
ax1.add_patch(db_box)
ax1.text(13.75, 9, '数据存储层', fontsize=12, fontweight='bold', ha='center', color=colors['database'])

db_modules = [('MySQL 数据库', 8), ('Redis 缓存', 7), ('文件存储', 6.2)]
for name, y in db_modules:
    box = FancyBboxPatch((12.7, y), 2.1, 0.6,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor='white', edgecolor=colors['database'], linewidth=1.5)
    ax1.add_patch(box)
    ax1.text(13.75, y+0.3, name, fontsize=10, ha='center', va='center', color='#333')

# --- 外部API (右下) ---
api_box = FancyBboxPatch((12.5, 2), 2.5, 3,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    facecolor=colors['api'], alpha=0.3, edgecolor=colors['api'], linewidth=2)
ax1.add_patch(api_box)
ax1.text(13.75, 4.5, '外部接口层', fontsize=12, fontweight='bold', ha='center', color=colors['api'])

api_modules = [('空气质量接口', 3.8), ('城市天气接口', 3), ('历史数据接口', 2.2)]
for name, y in api_modules:
    box = FancyBboxPatch((12.7, y), 2.1, 0.55,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor='white', edgecolor=colors['api'], linewidth=1.5)
    ax1.add_patch(box)
    ax1.text(13.75, y+0.28, name, fontsize=9, ha='center', va='center', color='#333')

# --- AI服务 (顶部) ---
ai_box = FancyBboxPatch((12.5, 9.5), 2.5, 2,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    facecolor=colors['ai'], alpha=0.3, edgecolor=colors['ai'], linewidth=2)
ax1.add_patch(ai_box)
ax1.text(13.75, 11, 'AI服务层', fontsize=12, fontweight='bold', ha='center', color='#996600')

ai_modules = [('MiniMax 大模型', 10.3), ('健康建议生成', 9.8)]
for name, y in ai_modules:
    box = FancyBboxPatch((12.7, y), 2.1, 0.5,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor='white', edgecolor=colors['ai'], linewidth=1.5)
    ax1.add_patch(box)
    ax1.text(13.75, y+0.25, name, fontsize=9, ha='center', va='center', color='#333')

# --- 连接线 (箭头) ---
# 前端 -> 后端
arrow_props = dict(arrowstyle='->', color='#333', lw=2)
ax1.annotate('', xy=(6.5, 7.5), xytext=(4, 7.5), arrowprops=arrow_props)
ax1.annotate('', xy=(6.5, 5), xytext=(4, 5), arrowprops=arrow_props)

# 后端 -> 数据库
ax1.annotate('', xy=(12.5, 8), xytext=(10.5, 8), arrowprops=arrow_props)
ax1.annotate('', xy=(12.5, 7), xytext=(10.5, 7), arrowprops=arrow_props)

# 后端 -> 外部API
ax1.annotate('', xy=(12.5, 4.5), xytext=(10.5, 5), arrowprops=dict(arrowstyle='->', color=colors['api'], lw=1.5, linestyle='dashed'))
ax1.annotate('', xy=(12.5, 3.5), xytext=(10.5, 4), arrowprops=dict(arrowstyle='->', color=colors['api'], lw=1.5, linestyle='dashed'))

# 后端 -> AI
ax1.annotate('', xy=(12.5, 10.5), xytext=(10.5, 8.5), arrowprops=dict(arrowstyle='->', color=colors['ai'], lw=1.5, linestyle='dashed'))

# 标注文字
ax1.text(5.2, 7.8, 'HTTP请求', fontsize=9, color='#666')
ax1.text(5.2, 5.3, 'JSON数据', fontsize=9, color='#666')
ax1.text(11.3, 8.2, '查询/存储', fontsize=9, color='#666')

# 图例
legend_items = [
    mpatches.Patch(facecolor=colors['frontend'], alpha=0.3, label='前端展示层'),
    mpatches.Patch(facecolor=colors['backend'], alpha=0.3, label='后端服务层'),
    mpatches.Patch(facecolor=colors['database'], alpha=0.3, label='数据存储层'),
    mpatches.Patch(facecolor=colors['api'], alpha=0.3, label='外部接口层'),
    mpatches.Patch(facecolor=colors['ai'], alpha=0.3, label='AI服务层'),
]
ax1.legend(handles=legend_items, loc='lower left', fontsize=10, framealpha=0.9)

plt.tight_layout()
fig1.savefig('图表/系统功能模块图.png', dpi=150, bbox_inches='tight', facecolor='white')
print("已生成: 图表/系统功能模块图.png")

# ============ 图2: 数据请求流程图 ============
fig2, ax2 = plt.subplots(1, 1, figsize=(20, 14))
ax2.set_xlim(0, 20)
ax2.set_ylim(0, 14)
ax2.axis('off')
ax2.set_title('空气质量预测系统 - 数据请求流程图', fontsize=20, fontweight='bold', pad=20, color='#333')

# 定义流程节点
def draw_box(ax, x, y, w, h, text, color, text_color='#333', font_size=10):
    box = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.2",
        facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, fontsize=font_size, ha='center', va='center',
            color=text_color, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, color='#333', style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=2))

def draw_diamond(ax, x, y, w, h, text, color):
    diamond = plt.Polygon([(x+w/2, y+h), (x+w, y+h/2), (x+w/2, y), (x, y+h/2)],
                         facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
    ax.add_patch(diamond)
    ax.text(x+w/2, y+h/2, text, fontsize=8, ha='center', va='center', color='#333', fontweight='bold')

def draw_circle(ax, x, y, r, text, color):
    circle = plt.Circle((x, y), r, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, text, fontsize=8, ha='center', va='center', color='#333', fontweight='bold')

# ==================== 流程A: 实时数据请求 ====================
ax2.text(3, 13.3, '流程A: 实时数据请求', fontsize=13, fontweight='bold', color=colors['frontend'], ha='center')

# 客户端
draw_box(ax2, 0.5, 11, 2.5, 1.2, '用户浏览器\n(前端)', colors['frontend'], '#fff', 10)
# 请求箭头
draw_arrow(ax2, 3, 11.6, 4.5, 11.6, colors['frontend'])
ax2.text(3.5, 12, '1. GET /api/cities', fontsize=8, color='#666')

# Flask API
draw_box(ax2, 5, 10.5, 3, 1.5, 'FastAPI\n(后端)', colors['backend'], '#fff', 10)
draw_arrow(ax2, 6.5, 10.5, 6.5, 9.5, colors['backend'])
ax2.text(6.8, 9.9, '2. 查询请求', fontsize=8, color='#666')

# 数据库
draw_box(ax2, 5, 7.5, 3, 1.5, 'MySQL\n数据库', colors['database'], '#fff', 10)
draw_arrow(ax2, 6.5, 7.5, 6.5, 6.5, colors['database'])
ax2.text(7, 7, '3. SQL查询', fontsize=8, color='#666')

# 返回数据
draw_box(ax2, 5, 4.5, 3, 1.5, '城市数据\n列表', '#90EE90', '#333', 9)
draw_arrow(ax2, 6.5, 6.5, 6.5, 5.5, '#90EE90')
ax2.text(7, 6, '4. 查询结果', fontsize=8, color='#666')

# 处理数据
draw_box(ax2, 5, 2.5, 3, 1.5, '数据处理\n模块', colors['cache'], '#fff', 9)
draw_arrow(ax2, 6.5, 4.5, 6.5, 3.5, colors['cache'])
ax2.text(7, 4, '5. 格式化', fontsize=8, color='#666')

# 返回前端
draw_arrow(ax2, 8, 3.25, 9.5, 3.25, colors['frontend'])
ax2.text(8.5, 3.6, '6. JSON响应', fontsize=8, color='#666')

# 前端渲染
draw_box(ax2, 10, 2.5, 2.5, 1.2, '前端渲染\n地图/列表', colors['frontend'], '#fff', 9)

# ==================== 流程B: 城市预测请求 ====================
ax2.text(14, 13.3, '流程B: 城市预测请求', fontsize=13, fontweight='bold', color=colors['api'], ha='center')

# 客户端
draw_box(ax2, 11, 11, 2.5, 1.2, '用户选择\n城市', colors['frontend'], '#fff', 9)
draw_arrow(ax2, 13.5, 11.6, 15, 11.6, colors['frontend'])
ax2.text(13.8, 12, '1. 预测请求', fontsize=8, color='#666')

# 预测API
draw_box(ax2, 15.5, 10.5, 3, 1.5, '预测API\n/ai/predict', colors['backend'], '#fff', 9)
draw_arrow(ax2, 17, 10.5, 17, 9.5, colors['backend'])
ax2.text(17.3, 9.9, '2. 调用预测', fontsize=8, color='#666')

# 预测引擎
draw_box(ax2, 15.5, 7.5, 3, 1.5, 'LSTM预测\n引擎', '#FFD93D', '#333', 9)
draw_arrow(ax2, 17, 7.5, 17, 6.5, '#FFD93D')
ax2.text(17.3, 7, '3. 模型预测', fontsize=8, color='#666')

# AI建议
draw_box(ax2, 15.5, 4.5, 3, 1.5, 'AI健康\n建议生成', colors['ai'], '#333', 9)
draw_arrow(ax2, 17, 6.5, 17, 5.5, colors['ai'])
ax2.text(17.3, 6, '4. AI分析', fontsize=8, color='#666')

# 返回
draw_arrow(ax2, 17, 4.5, 17, 3.5, colors['frontend'])
draw_box(ax2, 15.5, 2.5, 3, 1.5, '预测结果\n+ 建议', '#90EE90', '#333', 9)
ax2.text(17.3, 4, '5. 预测数据', fontsize=8, color='#666')

# ==================== 流程C: 历史数据查询 ====================
ax2.text(3, 0.8, '流程C: 历史数据查询', fontsize=13, fontweight='bold', color=colors['database'], ha='center')

draw_box(ax2, 0.5, -0.8, 2, 1, '用户选择\n日期范围', colors['frontend'], '#fff', 8)
draw_arrow(ax2, 2.5, -0.3, 3.5, -0.3, colors['frontend'])
draw_box(ax2, 4, -0.8, 2.5, 1, 'API查询\n/history', colors['backend'], '#fff', 8)
draw_arrow(ax2, 6.5, -0.3, 7.5, -0.3, colors['database'])
draw_box(ax2, 8, -0.8, 2.5, 1, '数据库\n时间查询', colors['database'], '#fff', 8)
draw_arrow(ax2, 10.5, -0.3, 11.5, -0.3, '#90EE90')
draw_box(ax2, 11.5, -0.8, 2.5, 1, '数据可视化\n图表', colors['frontend'], '#fff', 8)

# 标注说明
ax2.text(0.5, -2, '说明: 用户请求历史数据时，后端根据日期范围从MySQL查询记录，\n'
         '支持AQI/PM2.5/PM10/SO2/NO2/CO/O3等多种指标的时间序列展示。',
         fontsize=9, color='#666', linespacing=1.5)

# 图例
legend_elements = [
    mpatches.Patch(facecolor=colors['frontend'], alpha=0.3, label='前端模块'),
    mpatches.Patch(facecolor=colors['backend'], alpha=0.3, label='后端服务'),
    mpatches.Patch(facecolor=colors['database'], alpha=0.3, label='数据库'),
    mpatches.Patch(facecolor='#FFD93D', alpha=0.3, label='AI/预测模块'),
    mpatches.Patch(facecolor='#90EE90', alpha=0.3, label='返回/展示'),
]
ax2.legend(handles=legend_elements, loc='right', fontsize=9, framealpha=0.9, bbox_to_anchor=(0.98, 0.3))

plt.tight_layout()
fig2.savefig('图表/系统流程图.png', dpi=150, bbox_inches='tight', facecolor='white')
print("已生成: 图表/系统流程图.png")

# ============ 图3: API接口详细流程图 ============
fig3, ax3 = plt.subplots(1, 1, figsize=(16, 12))
ax3.set_xlim(0, 16)
ax3.set_ylim(0, 12)
ax3.axis('off')
ax3.set_title('API接口数据流转详解', fontsize=18, fontweight='bold', pad=20, color='#333')

# 顶部：前端
ax3.text(8, 11.5, '前端应用', fontsize=14, fontweight='bold', ha='center', color=colors['frontend'])
frontend_apis = [
    ('地图视图', 5, 10.8),
    ('城市列表', 5, 10.2),
    ('预测界面', 8, 10.8),
    ('历史数据', 8, 10.2),
]
for name, x, y in frontend_apis:
    box = FancyBboxPatch((x-1.2, y-0.3), 2.4, 0.55,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor=colors['frontend'], alpha=0.3, edgecolor=colors['frontend'], linewidth=1.5)
    ax3.add_patch(box)
    ax3.text(x, y, name, fontsize=9, ha='center', va='center', color='#333')

# 箭头指向后端
for x in [6.2, 6.2, 9.2, 9.2]:
    ax3.annotate('', xy=(10, 8.5), xytext=(x, 10.5),
                arrowprops=dict(arrowstyle='->', color=colors['frontend'], lw=1.5))

# 后端
ax3.text(12, 8.5, '后端服务 (FastAPI)', fontsize=12, fontweight='bold', ha='center', color=colors['backend'])
backend_box = FancyBboxPatch((10, 5.5), 4, 5.5,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    facecolor=colors['backend'], alpha=0.15, edgecolor=colors['backend'], linewidth=1.5)
ax3.add_patch(backend_box)

# API列表
api_items = [
    ('GET /api/cities', '获取城市列表', 7.8),
    ('GET /api/provinces', '获取省份数据', 7),
    ('GET /api/cities/24h', '24小时数据', 6.2),
    ('GET /api/cities/predict', '空气质量预测', 5.4),
    ('GET /api/cities/history', '历史数据查询', 4.6),
]
for endpoint, desc, y in api_items:
    ax3.text(10.5, y, endpoint, fontsize=9, fontweight='bold', color=colors['backend'])
    ax3.text(12.5, y, desc, fontsize=8, color='#666')

# 数据处理模块
ax3.text(7.5, 8.5, '数据处理层', fontsize=12, fontweight='bold', ha='center', color=colors['cache'])
process_box = FancyBboxPatch((5.5, 5.5), 4, 5.5,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    facecolor=colors['cache'], alpha=0.15, edgecolor=colors['cache'], linewidth=1.5)
ax3.add_patch(process_box)

process_items = [
    ('数据清洗', '过滤无效数据', 7.8),
    ('数据转换', '格式标准化', 7),
    ('数据聚合', '统计计算', 6.2),
    ('数据缓存', 'Redis存储', 5.4),
    ('模型推理', 'LSTM预测', 4.6),
]
for name, desc, y in process_items:
    ax3.text(6, y, name, fontsize=9, fontweight='bold', color=colors['cache'])
    ax3.text(7.8, y, desc, fontsize=8, color='#666')

# 箭头：前端 -> 后端
ax3.annotate('', xy=(10, 8.8), xytext=(7.2, 10.5),
            arrowprops=dict(arrowstyle='->', color=colors['frontend'], lw=2))

# 箭头：后端 -> 处理
ax3.annotate('', xy=(9.5, 8.5), xytext=(9.5, 7.5),
            arrowprops=dict(arrowstyle='<->', color=colors['cache'], lw=2))

# 底部：数据存储
ax3.text(7.5, 4.5, '数据存储层', fontsize=12, fontweight='bold', ha='center', color=colors['database'])
storage_box = FancyBboxPatch((4, 1.5), 7, 2.5,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    facecolor=colors['database'], alpha=0.15, edgecolor=colors['database'], linewidth=1.5)
ax3.add_patch(storage_box)

storage_items = [
    ('MySQL', '城市数据/历史记录', 3),
    ('Redis', '缓存数据/API响应', 2.2),
    ('文件存储', '地图JSON/配置', 1.4),
]
for name, desc, y in storage_items:
    ax3.text(5, y, name, fontsize=10, fontweight='bold', color=colors['database'])
    ax3.text(6.8, y, desc, fontsize=8, color='#666')

# 箭头：处理 -> 存储
ax3.annotate('', xy=(6, 4.8), xytext=(7, 4.8),
            arrowprops=dict(arrowstyle='<->', color=colors['database'], lw=1.5))
ax3.annotate('', xy=(9, 4.8), xytext=(10, 4.8),
            arrowprops=dict(arrowstyle='<->', color=colors['database'], lw=1.5))

# API响应格式说明
ax3.text(0.5, 8.5, 'API响应格式示例:', fontsize=10, fontweight='bold', color='#333')
response_box = FancyBboxPatch((0.5, 4.5), 4, 3.8,
    boxstyle="round,pad=0.1,rounding_size=0.2",
    facecolor='#f8f9fa', edgecolor='#ddd', linewidth=1)
ax3.add_patch(response_box)

response_text = """{
  "code": 200,
  "data": [
    {
      "cityname": "北京",
      "province": "北京",
      "aqi": 85,
      "pm25": 58,
      "pm10": 92,
      "so2": 10,
      "no2": 45,
      "co": 0.8,
      "o3": 156
    }
  ],
  "message": "success"
}"""
ax3.text(0.8, 7.8, response_text, fontsize=7, family='monospace', color='#333', linespacing=1.3)

# 右侧：AI服务
ax3.text(14.5, 3, 'AI服务层', fontsize=11, fontweight='bold', ha='center', color=colors['ai'])
ai_box = FancyBboxPatch((13, 0.5), 3, 4,
    boxstyle="round,pad=0.1,rounding_size=0.3",
    facecolor=colors['ai'], alpha=0.15, edgecolor=colors['ai'], linewidth=1.5)
ax3.add_patch(ai_box)

ai_items = [
    ('健康建议', '基于指标', 3.5),
    ('数据分析', '趋势解读', 2.7),
    ('预测报告', '多指标评估', 1.9),
    ('智能推荐', '防护措施', 1.1),
]
for name, desc, y in ai_items:
    ax3.text(13.3, y, name, fontsize=9, fontweight='bold', color='#996600')
    ax3.text(14.5, y, desc, fontsize=7, color='#666')

# 连接
ax3.annotate('', xy=(13, 4), xytext=(10, 4),
            arrowprops=dict(arrowstyle='->', color=colors['ai'], lw=1, linestyle='dashed'))

plt.tight_layout()
fig3.savefig('图表/API接口数据流转图.png', dpi=150, bbox_inches='tight', facecolor='white')
print("已生成: 图表/API接口数据流转图.png")

# ============ 图4: 完整系统架构图 ============
fig4, ax4 = plt.subplots(1, 1, figsize=(18, 14))
ax4.set_xlim(0, 18)
ax4.set_ylim(0, 14)
ax4.axis('off')
ax4.set_title('空气质量预测系统 - 完整架构图', fontsize=20, fontweight='bold', pad=20, color='#333')

# 层次划分
layers = [
    (11, 13, '用户交互层', colors['frontend'], '#fff'),
    (8.5, 10.5, '前端服务层', '#45B7D1', '#fff'),
    (6, 8, '后端服务层', colors['backend'], '#fff'),
    (3.5, 5.5, '数据处理层', colors['cache'], '#fff'),
    (0.5, 3, '数据存储层', colors['database'], '#fff'),
]

for start, end, name, color, text_color in layers:
    rect = FancyBboxPatch((0.3, start), 17.4, end - start,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        facecolor=color, alpha=0.08, edgecolor=color, linewidth=1, linestyle='--')
    ax4.add_patch(rect)
    ax4.text(0.5, (start+end)/2, name, fontsize=11, fontweight='bold', color=color, va='center')

# 用户层
users = [('用户A\n浏览器', 2, 12.5), ('用户B\n移动端', 6, 12.5), ('用户C\n管理后台', 10, 12.5)]
for name, x, y in users:
    circle = plt.Circle((x, y), 0.6, facecolor=colors['frontend'], alpha=0.5, edgecolor=colors['frontend'], linewidth=2)
    ax4.add_patch(circle)
    ax4.text(x, y, name, fontsize=8, ha='center', va='center', color='#333')

# CDN/负载均衡
cdn_box = FancyBboxPatch((12, 11.5), 4, 1.2,
    boxstyle="round,pad=0.05,rounding_size=0.15",
    facecolor='#FF6B6B', alpha=0.3, edgecolor='#FF6B6B', linewidth=1.5)
ax4.add_patch(cdn_box)
ax4.text(14, 12.1, 'CDN / Nginx负载均衡', fontsize=9, ha='center', va='center', color='#333')

for x in [2.6, 6.6, 10.6]:
    ax4.annotate('', xy=(12.5, 12.1), xytext=(x, 12.1),
                arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.5))

# 前端组件
frontend_components = [
    ('ECharts地图', 13.5, 10.5),
    ('Vue.js框架', 15.5, 10.5),
    ('Ajax请求', 14.5, 9.8),
]
for name, x, y in frontend_components:
    box = FancyBboxPatch((x-1, y-0.25), 2, 0.5,
        boxstyle="round,pad=0.03,rounding_size=0.1",
        facecolor='white', edgecolor=colors['frontend'], linewidth=1)
    ax4.add_patch(box)
    ax4.text(x, y, name, fontsize=8, ha='center', va='center', color='#333')

# 后端服务
backend_services = [
    ('FastAPI\nREST API', 2, 7.5),
    ('预测引擎\nLSTM', 5, 7.5),
    ('数据处理\n模块', 8, 7.5),
    ('任务调度\nCelery', 11, 7.5),
]
for name, x, y in backend_services:
    box = FancyBboxPatch((x-1.2, y-0.5), 2.4, 1,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor='white', edgecolor=colors['backend'], linewidth=1.5)
    ax4.add_patch(box)
    ax4.text(x, y, name, fontsize=9, ha='center', va='center', color='#333', fontweight='bold')

# 数据处理
data_process = [
    ('数据清洗', 3, 5),
    ('特征工程', 6, 5),
    ('模型训练', 9, 5),
    ('结果评估', 12, 5),
]
for name, x, y in data_process:
    box = FancyBboxPatch((x-1, y-0.35), 2, 0.7,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor='white', edgecolor=colors['cache'], linewidth=1.5)
    ax4.add_patch(box)
    ax4.text(x, y, name, fontsize=8, ha='center', va='center', color='#333')

# 数据存储
databases = [
    ('MySQL\n城市数据', 3, 2),
    ('Redis\n缓存', 7, 2),
    ('MongoDB\n历史数据', 11, 2),
]
for name, x, y in databases:
    box = FancyBboxPatch((x-1.2, y-0.5), 2.4, 1,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor='white', edgecolor=colors['database'], linewidth=1.5)
    ax4.add_patch(box)
    ax4.text(x, y, name, fontsize=8, ha='center', va='center', color='#333', fontweight='bold')

# 外部服务 (右侧)
external_box = FancyBboxPatch((14.5, 3), 3, 6,
    boxstyle="round,pad=0.1,rounding_size=0.2",
    facecolor='#FF6B6B', alpha=0.1, edgecolor='#FF6B6B', linewidth=1.5)
ax4.add_patch(external_box)
ax4.text(16, 8.5, '外部服务', fontsize=10, fontweight='bold', ha='center', color='#FF6B6B')

externals = [
    ('空气质量\n数据接口', 15.5, 7.5),
    ('天气API', 16.5, 6.5),
    ('AI大模型\n(MiniMax)', 15.5, 5.5),
    ('地图服务', 16.5, 4.5),
    ('第三方\n数据源', 15.5, 3.5),
]
for name, x, y in externals:
    box = FancyBboxPatch((x-1, y-0.4), 2, 0.8,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor='white', edgecolor='#FF6B6B', linewidth=1)
    ax4.add_patch(box)
    ax4.text(x, y, name, fontsize=7, ha='center', va='center', color='#333')

# 连接线
connections = [
    (14, 12.1, 14, 11),   # CDN -> Nginx
    (2, 11.9, 2, 8.5),    # 用户 -> Flask
    (5, 11.9, 5, 8.5),
    (9, 11.9, 9, 8.5),
    (5, 7, 5, 5.5),       # 后端 -> 数据处理
    (8, 7, 8, 5.5),
    (11, 7, 11, 5.5),
    (5, 4.5, 5, 3.5),     # 数据处理 -> 存储
    (8, 4.5, 8, 3.5),
    (11, 4.5, 11, 3.5),
    (13, 7, 14.5, 7.5),   # 后端 -> 外部
    (13, 5.5, 14.5, 5.5),
    (13, 3.5, 14.5, 3.5),
]
for x1, y1, x2, y2 in connections:
    ax4.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1.2))

# 图例
legend_items = [
    mpatches.Patch(facecolor=colors['frontend'], alpha=0.3, label='用户交互层'),
    mpatches.Patch(facecolor='#45B7D1', alpha=0.3, label='前端服务层'),
    mpatches.Patch(facecolor=colors['backend'], alpha=0.3, label='后端服务层'),
    mpatches.Patch(facecolor=colors['cache'], alpha=0.3, label='数据处理层'),
    mpatches.Patch(facecolor=colors['database'], alpha=0.3, label='数据存储层'),
    mpatches.Patch(facecolor='#FF6B6B', alpha=0.3, label='外部服务'),
]
ax4.legend(handles=legend_items, loc='lower left', fontsize=9, framealpha=0.9)

plt.tight_layout()
fig4.savefig('图表/完整系统架构图.png', dpi=150, bbox_inches='tight', facecolor='white')
print("已生成: 图表/完整系统架构图.png")

print("\n所有图表已生成完毕！")
plt.show()