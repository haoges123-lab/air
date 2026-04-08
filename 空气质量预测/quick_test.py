import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from 数据集 import AirQualityDataset
from 省份空气质量模型 import AirQualityModel

ds_train = AirQualityDataset(seq_len=30, split="train")
ds_val = AirQualityDataset(seq_len=30, split="val")
print(f"Train: {len(ds_train)}, Val: {len(ds_val)}")
print(f"Scales sample (北京): {list(ds_train.scales['北京'].items())[:3]}")

train_dl = DataLoader(ds_train, batch_size=64, shuffle=True)

model = AirQualityModel(
    num_provinces=ds_train.num_provinces,
    seq_len=30,
    seq_input_dim=7,
    hidden_size=256,
    num_layers=2,
    dropout=0.2,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

model.train()
for epoch in range(3):
    total_loss = 0
    for pids, inputs, labels in train_dl:
        optimizer.zero_grad()
        pred = model(pids, inputs)
        loss = nn.MSELoss()(pred, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}: avg loss = {total_loss / len(train_dl):.6f}")

model.eval()
with torch.inference_mode():
    pids, inputs, labels = ds_val[0]
    pids = pids.unsqueeze(0)
    inputs = inputs.unsqueeze(0)
    pred = model(pids, inputs)
    print(f"\nVal sample 0 - pred: {pred[0, :3]}, label: {labels[:3]}")
