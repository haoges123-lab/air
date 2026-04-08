import json
import os
from collections import defaultdict


def process_province_data():
    input_dir = "data"
    output_dir = "处理后数据"

    os.makedirs(output_dir, exist_ok=True)

    indicators = ["aqi", "pm25", "pm10", "so2", "no2", "co", "o3"]

    for province_file in os.listdir(input_dir):
        if not province_file.endswith(".json"):
            continue

        province_name = province_file.replace(".json", "")
        print(f"处理 {province_name}...")

        with open(os.path.join(input_dir, province_file), "r", encoding="utf-8") as f:
            province_data = json.load(f)

        date_to_data = defaultdict(dict)

        for city_name, city_data in province_data.items():
            for indicator in indicators:
                indicator_data = city_data.get(indicator, {}).get("data", [])
                for item in indicator_data:
                    date = item["date"]
                    value = item["value"]
                    if date not in date_to_data:
                        date_to_data[date] = {"count": 0}
                    if indicator not in date_to_data[date]:
                        date_to_data[date][indicator] = []
                    date_to_data[date][indicator].append(value)
                    date_to_data[date]["count"] += 1

        processed_data = []
        for date in sorted(date_to_data.keys()):
            day_data = {"date": date}
            for indicator in indicators:
                values = date_to_data[date].get(indicator, [])
                if values:
                    day_data[indicator] = sum(values) / len(values)
                else:
                    day_data[indicator] = None
            day_data["city_count"] = date_to_data[date]["count"] // len(indicators)
            processed_data.append(day_data)

        output_file = os.path.join(output_dir, f"{province_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({province_name: processed_data}, f, ensure_ascii=False, indent=2)

        print(f"  {province_name}: {len(processed_data)}天数据")

    print(f"\n处理完成！数据保存在: {output_dir}")


if __name__ == "__main__":
    process_province_data()
