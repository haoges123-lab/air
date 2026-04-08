import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from 数据集 import AirQualityDataset, FEATURE_DIM
from 省份空气质量模型 import AirQualityModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
LR = 3e-4
SEQ_LEN = 30
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.2
EPOCH = 500
PATIENCE = 60
SAVE_DIR = "weights"
os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    ds_train = AirQualityDataset(seq_len=SEQ_LEN, split="train")
    ds_val = AirQualityDataset(seq_len=SEQ_LEN, split="val")

    print(f"训练样本数: {len(ds_train)}")
    print(f"验证样本数: {len(ds_val)}")
    print(f"省份数: {ds_train.num_provinces}")
    print(f"特征维度: {FEATURE_DIM}")

    train_dl = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AirQualityModel(
        num_provinces=ds_train.num_provinces,
        seq_len=SEQ_LEN,
        seq_input_dim=FEATURE_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCH, eta_min=1e-6
    )

    print(f"\n设备: {DEVICE}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

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
        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCH} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {current_lr:.2e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "scales": ds_train.scales,
                    "province_to_id": ds_train.province_to_id,
                    "province_names": ds_train.province_names,
                },
                os.path.join(SAVE_DIR, "best_model.pt"),
            )
            print(f"  保存最佳模型, Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n早停触发 (patience={PATIENCE}), 训练结束")
                break

    torch.save(
        {
            "model": model.state_dict(),
            "scales": ds_train.scales,
            "province_to_id": ds_train.province_to_id,
            "province_names": ds_train.province_names,
        },
        os.path.join(SAVE_DIR, "last_model.pt"),
    )
    print("\n训练完成！")


if __name__ == "__main__":
    main()
