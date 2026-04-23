FROM python:3.10-slim

WORKDIR /app

# 1. 设置全局 pip 源为清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple


RUN pip config set global.extra-index-url https://mirrors.aliyun.com/pypi/simple/

COPY requirements.txt .

# 3. 安装基础依赖
RUN pip install --no-cache-dir -r requirements.txt

# 4. 安装 PyTorch (指定 CPU 版)
# 注意：PyTorch 官方源通常较慢，但依赖包现在会走阿里云源
RUN pip install --no-cache-dir \
    torch==2.4.1+cpu \
    torchvision==0.19.1+cpu \
    torchaudio==2.4.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]