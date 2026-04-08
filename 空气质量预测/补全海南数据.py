import json
import os
import time
import urllib.request
import urllib.parse

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "http://152.136.239.96:3000/",
}


def fetch_city_data(city_name, start_date, end_date, retries=3, delay=2.0):
    url = f"http://152.136.239.96:3000/api/history/batch-air-quality?cityname={urllib.parse.quote(city_name)}&type=daily&start={start_date}&end={end_date}"
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                if data.get("code") == 200:
                    return data.get("data", {}).get("indicatorsData", {})
                else:
                    print(f"  [{city_name}] fail: {data.get('msg', 'unknown')}")
                    return None
        except Exception as e:
            if attempt < retries - 1:
                wait_time = delay * (attempt + 1)
                print(
                    f"  [{city_name}] retry {attempt + 1}/{retries}: {e}, waiting {wait_time}s"
                )
                time.sleep(wait_time)
            else:
                print(f"  [{city_name}] fail: {e}")
                return None
    return None


def main():
    missing_cities = ["海口", "三亚", "三沙", "儋州"]
    start = "2025-10-01"
    end = "2026-03-31"

    province_data = {}
    for city in missing_cities:
        print(f"fetching {city}...", end=" ", flush=True)
        city_data = fetch_city_data(city, start, end)
        if city_data:
            province_data[city] = city_data
            aqi_count = len(city_data.get("aqi", {}).get("data", []))
            print(f"OK ({aqi_count} aqi records)")
        else:
            print("FAIL")
        time.sleep(0.5)

    file_path = os.path.join("data", "海南.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(province_data, f, ensure_ascii=False, indent=2)
    print(f"\n海南 data saved to {file_path}")


if __name__ == "__main__":
    main()
