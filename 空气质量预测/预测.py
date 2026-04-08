import os

import torch

from 数据集 import AirQualityDataset, INDICATORS, FEATURE_DIM
from 省份空气质量模型 import AirQualityModel


MODEL_INDICATORS = ["so2", "no2", "co", "o3"]


def predict(province_name, seq_data, model, province_to_id):
    province_id = province_to_id.get(province_name)
    if province_id is None:
        raise ValueError(f"未知省份: {province_name}")

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

    print(f"省份列表: {province_names}")
    print(f"验证样本数: {len(ds_val)}")
    print(f"模型预测指标: {MODEL_INDICATORS}")
    print(f"朴素基准指标: {[i for i in INDICATORS if i not in MODEL_INDICATORS]}")
    print()

    print("=" * 60)
    print("使用最后30天数据预测明天的空气质量 (验证集)")
    print("=" * 60)

    for i in range(min(5, len(ds_val))):
        province_id, inputs, labels = ds_val[-(i + 1)]
        province_name = province_names[province_id]

        pred = predict(province_name, inputs, model, ds_val.province_to_id)

        pred_unscaled = ds_val.unscale(torch.tensor(pred), province_name)
        label_unscaled = ds_val.unscale(labels, province_name)
        last_unscaled = ds_val.unscale(inputs[-1], province_name)

        final = []
        for j, ind in enumerate(INDICATORS):
            if ind in MODEL_INDICATORS:
                final.append(pred_unscaled[j])
            else:
                final.append(last_unscaled[j])

        model_vals = pred_unscaled
        baseline_vals = last_unscaled

        print(f"\n{province_name} - 样本{i + 1}")
        print("-" * 60)
        print(f"{'指标':<8} {'预测值':>10} {'真实值':>10} {'误差':>10} {'误差率':>10}")
        print("-" * 60)
        for j, ind in enumerate(INDICATORS):
            error = abs(final[j] - label_unscaled[j])
            rate = error / (abs(label_unscaled[j]) + 1e-8) * 100
            src = "模型" if ind in MODEL_INDICATORS else "朴素"
            print(
                f"{ind:<8} {final[j]:>10.2f} {label_unscaled[j]:>10.2f} "
                f"{error:>10.2f} {rate:>9.1f}% [{src}]"
            )

    n = len(ds_val)
    print("\n" + "=" * 60)
    print(f"验证集平均误差 (全{n}样本)")
    print("-" * 60)

    all_errors = {ind: [] for ind in INDICATORS}

    for idx in range(n):
        province_id, inputs, labels = ds_val[idx]
        province_name = province_names[province_id]

        pred = predict(province_name, inputs, model, ds_val.province_to_id)
        pred_unscaled = ds_val.unscale(torch.tensor(pred), province_name)
        label_unscaled = ds_val.unscale(labels, province_name)
        last_unscaled = ds_val.unscale(inputs[-1], province_name)

        for j, ind in enumerate(INDICATORS):
            if ind in MODEL_INDICATORS:
                val = pred_unscaled[j]
            else:
                val = last_unscaled[j]
            all_errors[ind].append(abs(val - label_unscaled[j]))

    for ind in INDICATORS:
        avg = sum(all_errors[ind]) / len(all_errors[ind])
        ok = "[OK]" if avg < 10 else "[!!]"
        print(f"  {ind:<8} 平均绝对误差: {avg:>8.2f}  {ok}")


if __name__ == "__main__":
    main()
