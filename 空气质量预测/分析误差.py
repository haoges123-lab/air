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

print("验证集最后15天 vs 前137天 误差分布对比:")
for ind in INDICATORS:
    early = []
    late = []
    for province, days in all_data.items():
        max_start = len(days) - 30
        train_end = max_start - 15
        for i in range(max_start):
            lv = days[i + 29].get(ind)
            tv = days[i + 30].get(ind)
            if lv is None or tv is None:
                continue
            err = abs(lv - tv)
            if i < train_end:
                early.append(err)
            else:
                late.append(err)

    early_avg = sum(early) / len(early) if early else 0
    late_avg = sum(late) / len(late) if late else 0
    print(
        f"  {ind:<8} 训练期均值误差={early_avg:.1f}, 验证期均值误差={late_avg:.1f}, "
        f"验证期/训练期比={late_avg / (early_avg + 1e-8):.1f}x"
    )

print("\n验证集单日最大变化:")
for ind in INDICATORS:
    max_jump = 0
    for province, days in all_data.items():
        max_start = len(days) - 30
        train_end = max_start - 15
        for i in range(train_end, max_start):
            lv = days[i + 29].get(ind)
            tv = days[i + 30].get(ind)
            if lv is None or tv is None:
                continue
            err = abs(lv - tv)
            if err > max_jump:
                max_jump = err
    print(f"  {ind:<8} 最大单日变化={max_jump:.1f}")
