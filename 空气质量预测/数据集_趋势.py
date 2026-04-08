import json
import os
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
from 数据集 import AirQualityDataset, INDICATORS
from 省份空气质量模型 import AirQualityModel


INDICATORS = ["aqi", "pm25", "pm10", "so2", "no2", "co", "o3"]


class DeltaAirQualityDataset(Dataset):
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
                last_day = days_data[i + self.seq_len - 1]
                target_day = days_data[i + self.seq_len]

                inp_vals = []
                for day in inp:
                    row = []
                    for ind in INDICATORS:
                        v = day.get(ind)
                        mean, std = scales[ind]
                        row.append((v - mean) / std if v is not None else 0.0)
                    inp_vals.append(row)

                lab_vals = []
                for ind in INDICATORS:
                    lv = last_day.get(ind)
                    tv = target_day.get(ind)
                    mean, std = scales[ind]
                    if lv is not None and tv is not None:
                        delta = (tv - lv) / std
                        lab_vals.append(delta)
                    else:
                        lab_vals.append(0.0)

                self.inputs.append(torch.tensor(inp_vals, dtype=torch.float32))
                self.labels.append(torch.tensor(lab_vals, dtype=torch.float32))
                self.province_ids.append(pid)

    def unscale(self, delta_values, province_name, last_day_values):
        scales = self.scales[province_name]
        result = []
        for j, ind in enumerate(INDICATORS):
            mean, std = scales[ind]
            delta = delta_values[j].item()
            pred = last_day_values[j] + delta * std
            result.append(pred)
        return result

    def get_last_day(self, idx):
        return self.inputs[idx][-1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.province_ids[idx], self.inputs[idx], self.labels[idx]


if __name__ == "__main__":
    from torch.utils.data import Dataset

    print("DeltaAirQualityDataset loaded")
