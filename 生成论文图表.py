"""
论文图表生成脚本
"""

import os
import math
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============== 图3.2: 模型架构图 ==============
def plot_model_architecture():
    """Bi-LSTM空气质量预测模型架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('Bi-LSTM空气质量预测模型架构', fontsize=16, fontweight='bold', pad=20)

    # 输入层 - 左上
    ax.add_patch(plt.Rectangle((0.5, 5), 2.5, 1.2, facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
    ax.text(1.75, 5.6, '省份ID (整数)', ha='center', va='center', fontsize=10)

    # 嵌入层 - 左中
    ax.add_patch(plt.Rectangle((0.5, 3.5), 2.5, 1.2, facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2))
    ax.text(1.75, 4.1, '省份嵌入层 (16维)', ha='center', va='center', fontsize=10)

    # 历史序列 - 中上
    ax.add_patch(plt.Rectangle((4, 5), 2.5, 1.2, facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2))
    ax.text(5.25, 5.6, '历史序列 (30x17)', ha='center', va='center', fontsize=10)

    # 双向LSTM - 中中
    ax.add_patch(plt.Rectangle((4, 2.5), 2.5, 2, facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2))
    ax.text(5.25, 3.5, '双向LSTM', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5.25, 3.0, '(64单元 x 2方向)', ha='center', va='center', fontsize=9)

    # 拼接 - 右
    ax.add_patch(plt.Rectangle((7.5, 3.5), 2, 1.5, facecolor='#E0F7FA', edgecolor='#00796B', linewidth=2))
    ax.text(8.5, 4.25, '特征拼接\n(144维)', ha='center', va='center', fontsize=10)

    # 全连接 - 右
    ax.add_patch(plt.Rectangle((10, 3.2), 1.5, 2, facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2))
    ax.text(10.75, 4.2, '全连接层\n64 -> 7', ha='center', va='center', fontsize=9)

    # 输出
    ax.add_patch(plt.Rectangle((10, 1), 1.5, 1, facecolor='#D7CCC8', edgecolor='#5D4037', linewidth=2))
    ax.text(10.75, 1.5, '预测值 (7维)', ha='center', va='center', fontsize=9)

    # 箭头 - 直线不交叉
    arrows = [
        ((3, 5.6), (4, 5.6)),           # 省份ID -> 嵌入
        ((3, 4.1), (7.5, 4.5)),         # 嵌入 -> 拼接
        ((6.5, 5.6), (7.5, 4.8)),        # 序列 -> 拼接
        ((6.5, 3.5), (7.5, 4.2)),        # LSTM -> 拼接
        ((9.5, 4.25), (10, 4.2)),        # 拼接 -> FC
        ((10.75, 3.2), (10.75, 2)),      # FC -> 输出
    ]
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.5))

    # 层级标注
    ax.text(1.75, 6.7, '输入层', ha='center', fontsize=11, fontweight='bold', color='#1976D2')
    ax.text(5.25, 6.7, '特征提取层', ha='center', fontsize=11, fontweight='bold', color='#7B1FA2')
    ax.text(10.75, 6.7, '输出层', ha='center', fontsize=11, fontweight='bold', color='#C62828')

    plt.tight_layout()
    plt.savefig('图表/3.2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已生成: 3.2.png")


# ============== 图5.2: 系统架构图 ==============
def plot_system_architecture():
    """空气质量预测系统架构图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('空气质量预测系统架构图', fontsize=14, fontweight='bold', pad=15)

    # 用户层 - 最上层
    ax.add_patch(plt.Rectangle((5, 7.5), 4, 1, facecolor='#BBDEFB', edgecolor='#1565C0', linewidth=2))
    ax.text(7, 8, '用户浏览器', ha='center', va='center', fontsize=12, fontweight='bold')

    # 前端层
    ax.add_patch(plt.Rectangle((5, 5.5), 4, 1.5, facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
    ax.text(7, 6.25, 'Web前端 (HTML/CSS/JS)', ha='center', va='center', fontsize=11)
    ax.text(7, 5.8, 'ECharts 可视化', ha='center', va='center', fontsize=10)

    # API层
    ax.add_patch(plt.Rectangle((5, 3.5), 4, 1.5, facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2))
    ax.text(7, 4.25, 'FastAPI 服务层', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7, 3.8, 'RESTful API / 路由控制', ha='center', va='center', fontsize=9)

    # 下方三层 - 数据层
    # 模型
    ax.add_patch(plt.Rectangle((1, 1.5), 3, 1.5, facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2))
    ax.text(2.5, 2.25, '模型推理模块', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(2.5, 1.8, 'PyTorch / Bi-LSTM', ha='center', va='center', fontsize=9)

    # 数据缓存
    ax.add_patch(plt.Rectangle((5, 1.5), 3, 1.5, facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2))
    ax.text(6.5, 2.25, '数据缓存模块', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(6.5, 1.8, 'JSON / 内存缓存', ha='center', va='center', fontsize=9)

    # 标准化参数
    ax.add_patch(plt.Rectangle((9.5, 1.5), 3, 1.5, facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2))
    ax.text(11, 2.25, '标准化参数', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(11, 1.8, '省份映射 / 均值标准差', ha='center', va='center', fontsize=9)

    # 外部API - 最右侧
    ax.add_patch(plt.Rectangle((12.5, 3), 1.5, 2, facecolor='#ECEFF1', edgecolor='#455A64', linewidth=2))
    ax.text(13.25, 4, '外部API', ha='center', va='center', fontsize=9)
    ax.text(13.25, 3.5, '中国环境\n监测总站', ha='center', va='center', fontsize=8)

    # 垂直箭头 - 主流程
    ax.annotate('', xy=(7, 5.5), xytext=(7, 7.5),
               arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2))
    ax.annotate('', xy=(7, 3.5), xytext=(7, 5.5),
               arrowprops=dict(arrowstyle='->', color='#F57F17', lw=2))

    # API层向下
    ax.annotate('', xy=(2.5, 1.5), xytext=(2.5, 3.5),
               arrowprops=dict(arrowstyle='->', color='#388E3C', lw=1.5))
    ax.annotate('', xy=(6.5, 1.5), xytext=(6.5, 3.5),
               arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5))
    ax.annotate('', xy=(11, 1.5), xytext=(11, 3.5),
               arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5))

    # 外部API横向箭头
    ax.annotate('', xy=(9.5, 4), xytext=(12.5, 4),
               arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.5))

    # 层标注
    ax.text(0.3, 8.2, '展示层', ha='center', fontsize=10, fontweight='bold', color='#1565C0')
    ax.text(0.3, 6.5, '前端层', ha='center', fontsize=10, fontweight='bold', color='#1976D2')
    ax.text(0.3, 4.5, '服务层', ha='center', fontsize=10, fontweight='bold', color='#F57F17')
    ax.text(0.3, 2, '数据层', ha='center', fontsize=10, fontweight='bold', color='#388E3C')

    plt.tight_layout()
    plt.savefig('图表/5.2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已生成: 5.2.png")


# ============== 图5.3: 数据流程图 ==============
def plot_data_flow():
    """系统数据流程图 - 横向流程布局"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('系统数据流程图', fontsize=14, fontweight='bold', pad=15)

    # 外部API
    ax.add_patch(plt.Rectangle((0.3, 3), 2, 1.5, facecolor='#ECEFF1', edgecolor='#455A64', linewidth=2))
    ax.text(1.3, 3.75, '外部API', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1.3, 3.25, '中国环境监测总站', ha='center', va='center', fontsize=9)

    # FastAPI后端
    ax.add_patch(plt.Rectangle((3.5, 2.5), 2.5, 2, facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
    ax.text(4.75, 4, 'FastAPI后端', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(4.75, 3.4, '数据获取', ha='center', va='center', fontsize=9)
    ax.text(4.75, 2.9, '缓存管理 / 模型推理', ha='center', va='center', fontsize=9)

    # Bi-LSTM模型
    ax.add_patch(plt.Rectangle((7.2, 3), 2, 1.5, facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2))
    ax.text(8.2, 3.75, 'Bi-LSTM', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(8.2, 3.25, '预测模型', ha='center', va='center', fontsize=9)

    # Web前端
    ax.add_patch(plt.Rectangle((10.5, 2.5), 2.5, 2, facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2))
    ax.text(11.75, 4, 'Web前端', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(11.75, 3.4, 'ECharts图表', ha='center', va='center', fontsize=9)
    ax.text(11.75, 2.9, '地图可视化 / 用户交互', ha='center', va='center', fontsize=8)

    # 下方数据存储
    ax.add_patch(plt.Rectangle((3.5, 0.5), 2.5, 1.2, facecolor='#BBDEFB', edgecolor='#1565C0', linewidth=1.5))
    ax.text(4.75, 1.1, '实时数据缓存', ha='center', va='center', fontsize=10)

    ax.add_patch(plt.Rectangle((6.5, 0.5), 2.5, 1.2, facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=1.5))
    ax.text(7.75, 1.1, '历史数据存储', ha='center', va='center', fontsize=10)

    ax.add_patch(plt.Rectangle((9.5, 0.5), 2, 1.2, facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=1.5))
    ax.text(10.5, 1.1, '模型权重', ha='center', va='center', fontsize=10)

    # 主流程箭头 - 水平
    ax.annotate('', xy=(3.5, 3.75), xytext=(2.3, 3.75),
               arrowprops=dict(arrowstyle='->', color='#455A64', lw=2))
    ax.annotate('', xy=(7.2, 3.75), xytext=(6, 3.75),
               arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2))
    ax.annotate('', xy=(10.5, 3.75), xytext=(9.2, 3.75),
               arrowprops=dict(arrowstyle='->', color='#388E3C', lw=2))

    # 数据存储连接 - 垂直虚线
    ax.annotate('', xy=(4.75, 1.7), xytext=(4.75, 2.5),
               arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=1, ls='--'))
    ax.annotate('', xy=(7.75, 1.7), xytext=(7.75, 3),
               arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=1, ls='--'))
    ax.annotate('', xy=(10.5, 1.7), xytext=(10.5, 3),
               arrowprops=dict(arrowstyle='->', color='#90A4AE', lw=1, ls='--'))

    # 标注
    ax.text(2, 5.5, '实时数据流', ha='center', fontsize=10, color='#1976D2', fontweight='bold')
    ax.text(5, 5.5, 'API请求与处理', ha='center', fontsize=10, color='#1976D2', fontweight='bold')
    ax.text(8.5, 5.5, '模型预测', ha='center', fontsize=10, color='#388E3C', fontweight='bold')
    ax.text(12, 5.5, '结果展示', ha='center', fontsize=10, color='#F57F17', fontweight='bold')

    ax.text(5.5, 0.1, '数据存储层', ha='center', fontsize=10, color='#7B1FA2', fontweight='bold')

    plt.tight_layout()
    plt.savefig('图表/5.3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已生成: 5.3.png")


# ============== 图3.4: 模型融合策略流程图 ==============
def plot_fusion_strategy():
    """模型融合预测策略流程图 - 清晰横向布局"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('模型融合预测策略流程图', fontsize=14, fontweight='bold', pad=15)

    # 输入
    ax.add_patch(plt.Rectangle((0.5, 2.2), 2, 1.5, facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))
    ax.text(1.5, 2.95, '输入数据', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(1.5, 2.45, '(30天历史序列)', ha='center', va='center', fontsize=9)

    # 分类
    ax.add_patch(plt.Rectangle((3.5, 2.2), 2, 1.5, facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2))
    ax.text(4.5, 2.95, '污染物', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(4.5, 2.45, '分类判断', ha='center', va='center', fontsize=9)

    # 上分支 - LSTM
    ax.add_patch(plt.Rectangle((6.5, 3.8), 2.5, 1.2, facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2))
    ax.text(7.75, 4.4, 'SO2 / NO2 / CO / O3', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.75, 3.9, '神经网络预测', ha='center', va='center', fontsize=9)

    # 下分支 - 朴素
    ax.add_patch(plt.Rectangle((6.5, 0.8), 2.5, 1.2, facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2))
    ax.text(7.75, 1.4, 'PM2.5 / PM10 / AQI', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(7.75, 0.9, '朴素基准法', ha='center', va='center', fontsize=9)

    # 融合
    ax.add_patch(plt.Rectangle((10, 2.2), 2, 1.5, facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2))
    ax.text(11, 2.95, '结果融合', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(11, 2.45, '(加权组合)', ha='center', va='center', fontsize=9)

    # 输出
    ax.add_patch(plt.Rectangle((12.5, 2.2), 1.5, 1.5, facecolor='#E0F7FA', edgecolor='#00796B', linewidth=2))
    ax.text(13.25, 2.95, '最终', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(13.25, 2.45, '预测值', ha='center', va='center', fontsize=9)

    # 箭头 - 水平主线
    ax.annotate('', xy=(3.5, 2.95), xytext=(2.5, 2.95),
               arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.5))

    # 分类后分叉
    ax.annotate('', xy=(7.75, 4.4), xytext=(5.5, 3.7),
               arrowprops=dict(arrowstyle='->', color='#388E3C', lw=1.5))
    ax.annotate('', xy=(7.75, 1.4), xytext=(5.5, 2.2),
               arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))

    # 汇聚
    ax.annotate('', xy=(10, 2.95), xytext=(9, 4.4),
               arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.5))
    ax.annotate('', xy=(10, 2.95), xytext=(9, 1.4),
               arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.5))

    # 输出
    ax.annotate('', xy=(12.5, 2.95), xytext=(12, 2.95),
               arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.5))

    # 分支说明
    ax.text(7.75, 5.2, 'LSTM预测 (alpha=1.0)', ha='center', fontsize=9, color='#388E3C', fontweight='bold')
    ax.text(7.75, 0.3, '朴素基准 (alpha=0.0)', ha='center', fontsize=9, color='#C62828', fontweight='bold')

    ax.text(5.8, 4.7, '预测难度大', ha='center', fontsize=8, color='#388E3C')
    ax.text(5.8, 0.3, '持续性强', ha='center', fontsize=8, color='#C62828')

    plt.tight_layout()
    plt.savefig('图表/3.4.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已生成: 3.4.png")


if __name__ == "__main__":
    os.makedirs('图表', exist_ok=True)

    # 只生成流程图相关的
    plot_model_architecture()
    plot_fusion_strategy()
    plot_system_architecture()
    plot_data_flow()

    print("\n" + "="*50)
    print("流程图图表生成完成！")
    print("="*50)