import json
import os
import math
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset


INDICATORS = ["aqi", "pm25", "pm10", "so2", "no2", "co", "o3"]
SEQ_LEN = 30
FEATURE_DIM = 7 + 10


def get_time_features(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    day_of_year = dt.timetuple().tm_yday
    month = dt.month
    week = dt.isocalendar()[1]
    day_of_week = dt.weekday()
    
    # 周期性特征
    sin_doy = math.sin(2 * math.pi * day_of_year / 365)
    cos_doy = math.cos(2 * math.pi * day_of_year / 365)
    sin_dow = math.sin(2 * math.pi * day_of_week / 7)
    cos_dow = math.cos(2 * math.pi * day_of_week / 7)
    sin_month = math.sin(2 * math.pi * month / 12)
    cos_month = math.cos(2 * math.pi * month / 12)
    
    # 季节特征（1-4分别代表春夏秋冬）
    season = (month % 12 + 3) // 3
    season_onehot = [0.0] * 4
    season_onehot[season - 1] = 1.0
    
    return [sin_doy, cos_doy, sin_dow, cos_dow, sin_month, cos_month] + season_onehot


class AirQualityDataset(Dataset):
    def __init__(self, data_dir="处理后数据", seq_len=30, split="train"):
        super().__init__()

        self.seq_len = seq_len
        self.split = split
        self.inputs = []
        self.labels = []
        self.province_ids = []
        self.scales = {}

        province_files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".json")]
        )
        self.province_names = [f.replace(".json", "") for f in province_files]
        self.province_to_id = {name: i for i, name in enumerate(self.province_names)}

        all_data = {}
        for file_name in province_files:
            province_name = file_name.replace(".json", "")
            with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as f:
                data = json.load(f)
            days_data = data[province_name]
            days_data.sort(key=lambda x: x["date"])
            all_data[province_name] = days_data

        self._build_features(all_data)

        self.num_provinces = len(self.province_names)

    def _build_features(self, all_data):
        for province_name, days_data in all_data.items():
            max_start = len(days_data) - self.seq_len
            train_end = max_start - 15

            ind_sums = {ind: 0.0 for ind in INDICATORS}
            ind_counts = {ind: 0 for ind in INDICATORS}
            ind_sqs = {ind: 0.0 for ind in INDICATORS}

            for i in range(max_start):
                if self.split == "train" and i >= train_end:
                    break
                if self.split == "val" and i < train_end:
                    continue
                for j in range(self.seq_len + 1):
                    day = days_data[i + j]
                    for ind in INDICATORS:
                        v = day.get(ind)
                        if v is not None:
                            ind_sums[ind] += v
                            ind_sqs[ind] += v * v
                            ind_counts[ind] += 1

            scales = {}
            for ind in INDICATORS:
                count = ind_counts[ind]
                if count > 0:
                    mean = ind_sums[ind] / count
                    var = max(ind_sqs[ind] / count - mean * mean, 0)
                    std = max(var**0.5, 1e-6)
                else:
                    mean, std = 0.0, 1.0
                scales[ind] = (mean, std)
            self.scales[province_name] = scales

        for province_name, days_data in all_data.items():
            pid = self.province_to_id[province_name]
            scales = self.scales[province_name]
            max_start = len(days_data) - self.seq_len
            train_end = max_start - 15
            for i in range(max_start):
                if self.split == "train" and i >= train_end:
                    break
                if self.split == "val" and i < train_end:
                    continue
                inp = days_data[i : i + self.seq_len]
                lab = days_data[i + self.seq_len]

                inp_vals = []
                for day in inp:
                    row = []
                    for ind in INDICATORS:
                        v = day.get(ind)
                        mean, std = scales[ind]
                        row.append((v - mean) / std if v is not None else 0.0)
                    time_feat = get_time_features(day["date"])
                    row.extend(time_feat)
                    inp_vals.append(row)

                lab_vals = []
                for ind in INDICATORS:
                    v = lab.get(ind)
                    mean, std = scales[ind]
                    lab_vals.append((v - mean) / std if v is not None else 0.0)

                self.inputs.append(torch.tensor(inp_vals, dtype=torch.float32))
                self.labels.append(torch.tensor(lab_vals, dtype=torch.float32))
                self.province_ids.append(pid)

    def unscale(self, values, province_name=None):
        if province_name is None:
            province_name = self.province_names[0]
        result = []
        for j, ind in enumerate(INDICATORS):
            mean, std = self.scales[province_name][ind]
            result.append(values[j].item() * std + mean)
        return result

    def get_scale(self, ind, province_name=None):
        if province_name is None:
            province_name = self.province_names[0]
        return self.scales[province_name][ind]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.province_ids[idx], self.inputs[idx], self.labels[idx]


if __name__ == "__main__":
    ds_train = AirQualityDataset(seq_len=SEQ_LEN, split="train")
    ds_val = AirQualityDataset(seq_len=SEQ_LEN, split="val")
    print(f"训练样本数: {len(ds_train)}")
    print(f"验证样本数: {len(ds_val)}")
    p, inp, lab = ds_train[0]
    print(f"输入形状: {inp.shape} (7指标 + 4时间特征)")
    print(f"标签形状: {lab.shape}")
