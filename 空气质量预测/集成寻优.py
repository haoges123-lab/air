import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from 数据集 import AirQualityDataset, INDICATORS
from 省份空气质量模型 import AirQualityModel


def predict(province_name, seq_data, model, province_to_id):
    province_id = province_to_id.get(province_name)
    model.eval()
    with torch.inference_mode():
        province_ids = torch.tensor([province_id])
        inputs = seq_data.unsqueeze(0)
        pred = model(province_ids, inputs).squeeze()
    return pred.numpy()


def main():
    ds_val = AirQualityDataset(seq_len=30, split="val")
    checkpoint = torch.load("weights/best_model.pt", weights_only=True)
    ds_val.scales = checkpoint["scales"]
    province_names = checkpoint["province_names"]

    model = AirQualityModel(
        num_provinces=ds_val.num_provinces,
        seq_len=30,
        seq_input_dim=7,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to("cpu")
    model.eval()

    n = len(ds_val)
    print(f"全验证集({n}样本) 集成方案寻优:\n")

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        abs_errors = {ind: [] for ind in INDICATORS}
        for idx in range(n):
            province_id, inputs, labels = ds_val[idx]
            province_name = province_names[province_id]

            pred = predict(province_name, inputs, model, ds_val.province_to_id)
            pred_unscaled = ds_val.unscale(torch.tensor(pred))
            label_unscaled = ds_val.unscale(labels)
            last_unscaled = ds_val.unscale(inputs[-1])

            for j, ind in enumerate(INDICATORS):
                blend = alpha * pred_unscaled[j] + (1 - alpha) * last_unscaled[j]
                abs_errors[ind].append(abs(blend - label_unscaled[j]))

        total_err = sum(sum(abs_errors[ind]) for ind in INDICATORS)
        print(f"alpha={alpha:.1f}: 总误差={total_err:.2f}")


if __name__ == "__main__":
    main()
