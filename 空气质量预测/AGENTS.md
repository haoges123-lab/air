# 智能体编码规范

## 项目概述

基于 PyTorch LSTM 的空气质量预测项目。根据历史数据预测中国各省份的空气质量指标（AQI、PM2.5、PM10、SO2、NO2、CO、O3）。

**依赖：** PyTorch，标准库（json、os、time、urllib、collections、math）

## 项目结构

| 文件 | 用途 |
|---|---|
| `获取空气质量数据.py` | 从 API 获取空气质量数据，保存为按省份划分的 JSON |
| `数据预处理.py` | 将城市级数据聚合为省级日均值 |
| `数据集.py` | PyTorch Dataset，滑动窗口序列（30天输入，1天标签） |
| `省份空气质量模型.py` | 带省份嵌入的 BiLSTM 模型 |
| `训练.py` | 训练/验证循环，保存最佳和最新模型检查点 |
| `预测.py` | 加载检查点，运行推理并报告误差 |

**数据目录：** `data/`（原始数据）、`处理后数据/`（聚合后数据）、`weights/`（模型检查点）

## 运行命令

```bash
# 获取数据（需要 API http://152.136.239.96:3000）
python 获取空气质量数据.py

# 预处理原始数据
python 数据预处理.py

# 训练模型（保存 best_model.pt 和 last_model.pt 到 weights/）
python 训练.py

# 运行预测演示（使用数据集最后5个样本）
python 预测.py
```

无正式测试套件。通过直接运行相关脚本验证修改。

## 代码风格

### Python 版本
需要 Python 3.8+。主要使用 `torch` 作为机器学习框架。

### 导入顺序
- 标准库在前，第三方库（`torch`、`torch.nn`、`torch.utils.data`）在后
- 不使用 star imports（如 `from torch import *`）
- 使用绝对导入（如 `from 数据集 import AirQualityDataset`），因为脚本从项目根目录运行
- `if __name__ == "__main__"` 的导入放在文件底部，避免循环依赖

### 格式化
- 4 空格缩进，不使用 Tab
- 运算符周围使用一致的空格
- 最大行长：120 字符（软限制）
- 顶级定义之间空 2 行，函数内部空 1 行

### 类型注解
- 函数参数和返回值在非显而易见时使用类型注解
- 张量使用 `torch.float32`
- 基础类型：`int`、`float`、`str`

### 命名规范

| 元素 | 规范 | 示例 |
|---|---|---|
| 模块/文件 | 中文（保留） | `数据预处理.py` |
| 类名 | PascalCase | `AirQualityDataset` |
| 函数名 | snake_case（允许中文） | `fetch_city_data`、`process_province_data` |
| 常量 | SCREAMING_SNAKE_CASE | `HEADERS`、`INDICATORS`、`SEQ_LEN` |
| 变量 | snake_case 或中文 | `province_data`、`city_name`、`province_embed` |
| 类型别名 | PascalCase | （暂未使用） |

### 错误处理
- 对用户可见的脚本使用宽泛的 `try/except Exception as e`，配合描述性的中文打印消息
- 不静默失败，始终打印或记录错误上下文
- `fetch_city_data` 使用指数退避重试逻辑处理网络错误
- 优先显式 `return None`，而非抛出异常

### PyTorch 模式
- 使用 `torch.inference_mode()`（而非已弃用的 `torch.no_grad()`）
- 显式使用 `.to(DEVICE)` 将张量移到设备
- 保存检查点用 `torch.save(model.state_dict(), ...)`
- 加载时使用 `weights_only=True`
- 模型输入输出使用 `torch.tensor(..., dtype=torch.float32)`

### 数据处理
- JSON：始终使用 `encoding="utf-8"`、`ensure_ascii=False`（支持中文字符）
- 路径：使用 `os.path.join` 保证跨平台兼容
- 创建输出目录用 `os.makedirs(path, exist_ok=True)`

### 关键常量

```python
INDICATORS = ["aqi", "pm25", "pm10", "so2", "no2", "co", "o3"]  # 7个空气质量指标
SEQ_LEN = 30          # 输入序列长度（天）
SEQ_INPUT_DIM = 7     # 每个指标每天一个值
HIDDEN_SIZE = 128     # LSTM 隐藏层大小
NUM_LAYERS = 2        # LSTM 层数
BIDIRECTIONAL = True  # 双向 LSTM
DROPOUT = 0.3
```

## 新增脚本

1. 放在项目根目录
2. 添加 `if __name__ == "__main__"` 块以便独立执行
3. 按顺序导入：标准库 → 第三方库 → 本地模块（分组，不交叉）
4. 用户面向输出使用中文，代码符号使用英文
5. 若有新的约定或依赖，在此文件中更新记录
