import urllib.request
import urllib.parse
import json

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "http://152.136.239.96:3000/",
}

cities = ["北京", "上海", "广州", "深圳", "海口", "三亚"]
for city in cities:
    url = f"http://152.136.239.96:3000/api/history/batch-air-quality?cityname={urllib.parse.quote(city)}&type=daily&start=2025-10-01&end=2025-10-05"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            aqi = (
                data.get("data", {})
                .get("indicatorsData", {})
                .get("aqi", {})
                .get("data", [])
            )
            print(f"{city}: code={data.get('code')}, aqi={len(aqi)} records")
    except Exception as e:
        print(f"{city}: FAIL - {e}")
