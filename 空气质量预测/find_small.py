import json
import os
import glob

files = sorted(glob.glob("处理后数据/*.json"))
for f in files:
    size = os.path.getsize(f)
    if size < 20000:
        print(f"small file: {os.path.basename(f)} = {size} bytes")
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        province = list(data.keys())[0]
        records = data[province]
        print(f"province: {province}")
        print(f"records: {len(records)}")
        print(f"first: {records[0]['date']}")
        print(f"last: {records[-1]['date']}")
        for r in records[:3]:
            print(r)
        print("...")
        for r in records[-3:]:
            print(r)
