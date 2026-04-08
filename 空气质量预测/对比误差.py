import os
import json
import torch
from 数据集 import AirQualityDataset, INDICATORS, FEATURE_DIM
from 省份空气质量模型 import AirQualityModel


def main():
    ds_val = AirQualityDataset(seq_len=30, split="val")
    checkpoint = torch.load("weights/best_model.pt", weights_only=True)
    ds_val.scales = checkpoint["scales"]
    ds_val.province_to_id = checkpoint["province_to_id"]
    province_names = checkpoint["province_names"]

    model = AirQualityModel(
        num_provinces=ds_val.num_provinces,
        seq_len=30,
        seq_input_dim=FEATURE_DIM,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to("cpu")
    model.eval()

    n = len(ds_val)

    model_errors = {ind: [] for ind in INDICATORS}
    baseline_errors = {ind: [] for ind in INDICATORS}

    with torch.inference_mode():
        for idx in range(n):
            province_id, inputs, labels = ds_val[idx]
            province_name = province_names[province_id]

            pred = (
                model(torch.tensor([province_id]), inputs.unsqueeze(0))
                .squeeze()
                .numpy()
            )
            pred_unscaled = ds_val.unscale(torch.tensor(pred))
            label_unscaled = ds_val.unscale(labels)
            last_unscaled = ds_val.unscale(inputs[-1])

            for j, ind in enumerate(INDICATORS):
                me = abs(pred_unscaled[j] - label_unscaled[j])
                be = abs(last_unscaled[j] - label_unscaled[j])
                model_errors[ind].append(me)
                baseline_errors[ind].append(be)

    print(f"{'指标':<8} {'模型均值':>10} {'朴素均值':>10} {'变化':>8}")
    print("-" * 40)
    for ind in INDICATORS:
        me_avg = sum(model_errors[ind]) / len(model_errors[ind])
        be_avg = sum(baseline_errors[ind]) / len(baseline_errors[ind])
        imp = (be_avg - me_avg) / be_avg * 100
        sym = "改善" if imp > 0 else "恶化"
        print(f"{ind:<8} {me_avg:>10.2f} {be_avg:>10.2f} {sym}{abs(imp):.1f}%")


if __name__ == "__main__":
    main()
