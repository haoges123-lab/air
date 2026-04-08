import json
import time
import urllib.request
import http.cookiejar

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Referer": "http://152.136.239.96:3000/",
    "Origin": "http://152.136.239.96:3000",
    "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}


def fetch(city, retries=5, base_delay=3.0):
    url = (
        "http://152.136.239.96:3000/api/history/batch-air-quality?cityname="
        + urllib.request.quote(city)
        + "&type=daily&start=2025-10-01&end=2026-03-31"
    )
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with opener.open(req, timeout=30) as resp:
                raw = resp.read()
                try:
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    import zlib

                    data = json.loads(zlib.decompress(raw).decode("utf-8"))
                if data.get("code") == 200:
                    return data.get("data", {}).get("indicatorsData", {})
                else:
                    print(f"  [{city}] 请求失败: {data.get('msg')}")
                    return None
        except Exception as e:
            if attempt < retries - 1:
                wait = base_delay * (2**attempt)
                print(f"  [{city}] 重试 {attempt + 1}/{retries}: {e}, 等待 {wait:.0f}s")
                time.sleep(wait)
            else:
                print(f"  [{city}] 失败: {e}")
                return None
    return None


def main():
    cities = ["西宁", "海东", "海北", "黄南", "海南", "果洛", "玉树", "海西"]
    print("开始获取青海各城市数据...")

    try:
        opener.open("http://152.136.239.96:3000/", timeout=10)
        time.sleep(1)
    except Exception:
        pass

    province_data = {}
    for city in cities:
        print(f"获取 {city}...", end=" ", flush=True)
        d = fetch(city)
        if d:
            cnt = len(d.get("aqi", {}).get("data", []))
            print(f"成功 ({cnt}条)")
            province_data[city] = d
        else:
            province_data[city] = {}
        time.sleep(2)

    with open("data/青海.json", "w", encoding="utf-8") as f:
        json.dump(province_data, f, ensure_ascii=False, indent=2)
    print("\n青海省数据已保存")


if __name__ == "__main__":
    main()
