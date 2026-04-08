import os
import json
from itertools import product

import torch
from torch import nn
from torch.utils.data import DataLoader

from 数据集 import AirQualityDataset, FEATURE_DIM
from 省份空气质量模型 import AirQualityModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "weights"
os.makedirs(SAVE_DIR, exist_ok=True)

# 超参数搜索空间
param_grid = {
    "hidden_size": [128, 256, 512],
    "num_layers": [2, 3],
    "dropout": [0.3, 0.4, 0.5],
    "lr": [5e-5, 1e-4, 3e-4],
    "batch_size": [32, 64, 128],
}

SEQ_LEN = 30
EPOCH = 100
PATIENCE = 30

def train_model(hidden_size, num_layers, dropout, lr, batch_size):
    """训练模型并返回验证损失"""
    print(f"\n训练配置: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}, lr={lr}, batch_size={batch_size}")
    
    ds_train = AirQualityDataset(seq_len=SEQ_LEN, split="train")
    ds_val = AirQualityDataset(seq_len=SEQ_LEN, split="val")
    
    train_dl = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = AirQualityModel(
        num_provinces=ds_train.num_provinces,
        seq_len=SEQ_LEN,
        seq_input_dim=FEATURE_DIM,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCH, eta_min=1e-6
    )
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(EPOCH):
        model.train()
        train_loss = 0
        for pids, inputs, labels in train_dl:
            pids = pids.to(DEVICE)
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(pids, inputs)
            loss = nn.HuberLoss(delta=1.0)(pred, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_dl)
        scheduler.step()
        
        model.eval()
        val_loss = 0
        with torch.inference_mode():
            for pids, inputs, labels in val_dl:
                pids = pids.to(DEVICE)
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                pred = model(pids, inputs)
                loss = nn.HuberLoss(delta=1.0)(pred, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_dl)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{EPOCH} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    return best_val_loss

def main():
    print("开始超参数搜索...")
    print(f"设备: {DEVICE}")
    print(f"超参数组合总数: {len(list(product(*param_grid.values())))}")
    
    results = []
    param_names = list(param_grid.keys())
    
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_names, params))
        val_loss = train_model(**param_dict)
        results.append((val_loss, param_dict))
    
    # 按验证损失排序
    results.sort(key=lambda x: x[0])
    
    print("\n超参数搜索结果:")
    for i, (val_loss, params) in enumerate(results[:5]):
        print(f"\n排名 {i+1}: 验证损失 = {val_loss:.4f}")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    # 保存最佳超参数
    best_loss, best_params = results[0]
    with open(os.path.join(SAVE_DIR, "best_hyperparams.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_loss": best_loss,
            "best_params": best_params
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n最佳超参数已保存到 {os.path.join(SAVE_DIR, 'best_hyperparams.json')}")

if __name__ == "__main__":
    main()
