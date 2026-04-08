import json
import os
import glob
from collections import defaultdict

INDICATORS = ["aqi", "pm25", "pm10", "so2", "no2", "co", "o3"]

all_data = {}
for f in sorted(glob.glob("处理后数据/*.json")):
    with open(f, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    name = list(data.keys())[0]
    all_data[name] = sorted(data[name], key=lambda x: x["date"])

total_error = defaultdict(float)
total_count = defaultdict(int)

for province, days in all_data.items():
    max_start = len(days) - 30
    train_end = max_start - 15
    for i in range(max_start):
        if i < train_end:
            continue
        last_day = days[i + 29]
        target_day = days[i + 30]
        for ind in INDICATORS:
            lv = last_day.get(ind)
            tv = target_day.get(ind)
            if lv is not None and tv is not None:
                err = abs(lv - tv)
                rate = err / (tv + 1e-8) * 100
                total_error[ind] += rate
                total_count[ind] += 1

print("朴素基准 (用最后一天预测明天):")
for ind in INDICATORS:
    avg = total_error[ind] / total_count[ind]
    print(f"  {ind:<8} 平均误差率: {avg:.1f}%")
