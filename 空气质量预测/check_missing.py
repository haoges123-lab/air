import json
from datetime import datetime, timedelta

with open("处理后数据/海南.json", "r", encoding="utf-8") as f:
    data = json.load(f)["海南"]

existing_dates = set(d["date"] for d in data)
start = datetime(2025, 10, 1)
end = datetime(2026, 3, 31)

missing = []
cur = start
while cur <= end:
    ds = cur.strftime("%Y-%m-%d")
    if ds not in existing_dates:
        missing.append(ds)
    cur += timedelta(days=1)

print(f"total existing: {len(data)}, missing: {len(missing)}")
print("missing dates:")
for d in missing:
    print(d)
