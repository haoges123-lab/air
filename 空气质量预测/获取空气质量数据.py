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

PROVINCES_CITIES = {
    "北京": ["北京"],
    "天津": ["天津"],
    "上海": ["上海"],
    "重庆": ["重庆"],
    "河北": [
        "石家庄",
        "唐山",
        "秦皇岛",
        "邯郸",
        "邢台",
        "保定",
        "张家口",
        "承德",
        "沧州",
        "廊坊",
        "衡水",
    ],
    "山西": [
        "太原",
        "大同",
        "阳泉",
        "长治",
        "晋城",
        "朔州",
        "晋中",
        "运城",
        "忻州",
        "临汾",
        "吕梁",
    ],
    "辽宁": [
        "沈阳",
        "大连",
        "鞍山",
        "抚顺",
        "本溪",
        "丹东",
        "锦州",
        "营口",
        "阜新",
        "辽阳",
        "盘锦",
        "铁岭",
        "朝阳",
        "葫芦岛",
    ],
    "吉林": ["长春", "吉林", "四平", "辽源", "通化", "白山", "松原", "白城"],
    "黑龙江": [
        "哈尔滨",
        "齐齐哈尔",
        "鸡西",
        "鹤岗",
        "双鸭山",
        "大庆",
        "伊春",
        "佳木斯",
        "七台河",
        "牡丹江",
        "黑河",
        "绥化",
    ],
    "江苏": [
        "南京",
        "无锡",
        "徐州",
        "常州",
        "苏州",
        "南通",
        "连云港",
        "淮安",
        "盐城",
        "扬州",
        "镇江",
        "泰州",
        "宿迁",
    ],
    "浙江": [
        "杭州",
        "宁波",
        "温州",
        "嘉兴",
        "湖州",
        "绍兴",
        "金华",
        "衢州",
        "舟山",
        "台州",
        "丽水",
    ],
    "安徽": [
        "合肥",
        "芜湖",
        "蚌埠",
        "淮南",
        "马鞍山",
        "淮北",
        "铜陵",
        "安庆",
        "黄山",
        "阜阳",
        "宿州",
        "滁州",
        "六安",
        "宣城",
        "池州",
        "亳州",
    ],
    "福建": ["福州", "厦门", "莆田", "三明", "泉州", "漳州", "南平", "龙岩", "宁德"],
    "江西": [
        "南昌",
        "景德镇",
        "萍乡",
        "九江",
        "新余",
        "鹰潭",
        "赣州",
        "吉安",
        "宜春",
        "抚州",
        "上饶",
    ],
    "山东": [
        "济南",
        "青岛",
        "淄博",
        "枣庄",
        "东营",
        "烟台",
        "潍坊",
        "济宁",
        "泰安",
        "威海",
        "日照",
        "临沂",
        "德州",
        "聊城",
        "滨州",
        "菏泽",
    ],
    "河南": [
        "郑州",
        "开封",
        "洛阳",
        "平顶山",
        "安阳",
        "鹤壁",
        "新乡",
        "焦作",
        "濮阳",
        "许昌",
        "漯河",
        "三门峡",
        "南阳",
        "商丘",
        "信阳",
        "周口",
        "驻马店",
    ],
    "湖北": [
        "武汉",
        "黄石",
        "十堰",
        "宜昌",
        "襄阳",
        "鄂州",
        "荆门",
        "孝感",
        "荆州",
        "黄冈",
        "咸宁",
        "随州",
        "恩施",
    ],
    "湖南": [
        "长沙",
        "株洲",
        "湘潭",
        "衡阳",
        "邵阳",
        "岳阳",
        "常德",
        "张家界",
        "益阳",
        "郴州",
        "永州",
        "怀化",
        "娄底",
        "湘西",
    ],
    "广东": [
        "广州",
        "韶关",
        "深圳",
        "珠海",
        "汕头",
        "佛山",
        "江门",
        "湛江",
        "茂名",
        "肇庆",
        "惠州",
        "梅州",
        "汕尾",
        "河源",
        "阳江",
        "清远",
        "东莞",
        "中山",
        "潮州",
        "揭州",
        "云浮",
    ],
    "海南": ["海口", "三亚", "三沙", "儋州"],
    "四川": [
        "成都",
        "自贡",
        "攀枝花",
        "泸州",
        "德阳",
        "绵阳",
        "广元",
        "遂宁",
        "内江",
        "乐山",
        "南充",
        "眉山",
        "宜宾",
        "广安",
        "达州",
        "雅安",
        "巴中",
        "资阳",
        "阿坝",
        "甘孜",
        "凉山",
    ],
    "贵州": [
        "贵阳",
        "六盘水",
        "遵义",
        "安顺",
        "毕节",
        "铜仁",
        "黔西南",
        "黔东南",
        "黔南",
    ],
    "云南": [
        "昆明",
        "曲靖",
        "玉溪",
        "保山",
        "昭通",
        "丽江",
        "普洱",
        "临沧",
        "楚雄",
        "红河",
        "文山",
        "西双版纳",
        "大理",
        "德宏",
        "怒江",
        "迪庆",
    ],
    "陕西": [
        "西安",
        "铜川",
        "宝鸡",
        "咸阳",
        "渭南",
        "延安",
        "汉中",
        "榆林",
        "安康",
        "商洛",
    ],
    "甘肃": [
        "兰州",
        "嘉峪关",
        "金昌",
        "白银",
        "天水",
        "武威",
        "张掖",
        "平凉",
        "酒泉",
        "庆阳",
        "定西",
        "陇南",
        "临夏",
        "甘南",
    ],
    "青海": ["西宁", "海东", "海北", "黄南", "海南", "果洛", "玉树", "海西"],
    "内蒙古": [
        "呼和浩特",
        "包头",
        "乌海",
        "赤峰",
        "通辽",
        "鄂尔多斯",
        "呼伦贝尔",
        "巴彦淖尔",
        "乌兰察布",
        "兴安",
        "锡林郭勒",
        "阿拉善",
    ],
    "广西": [
        "南宁",
        "柳州",
        "桂林",
        "梧州",
        "北海",
        "防城港",
        "钦州",
        "贵港",
        "玉林",
        "百色",
        "贺州",
        "河池",
        "来宾",
        "崇左",
    ],
    "西藏": ["拉萨", "日喀则", "昌都", "林芝", "山南", "那曲", "阿里"],
    "宁夏": ["银川", "石嘴山", "吴忠", "固原", "中卫"],
    "新疆": [
        "乌鲁木齐",
        "克拉玛依",
        "吐鲁番",
        "哈密",
        "昌吉",
        "博尔塔拉",
        "巴音郭楞",
        "阿克苏",
        "克孜勒苏",
        "喀什",
        "和田",
        "伊犁",
        "塔城",
        "阿勒泰",
    ],
}


def fetch_city_data(
    city_name: str, start_date: str, end_date: str, retries: int = 3, delay: float = 2.0
) -> dict:
    url = f"http://152.136.239.96:3000/api/history/batch-air-quality?cityname={urllib.parse.quote(city_name)}&type=daily&start={start_date}&end={end_date}"

    for attempt in range(retries):
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                if data.get("code") == 200:
                    return data.get("data", {}).get("indicatorsData", {})
                else:
                    print(f"  [{city_name}] 请求失败: {data.get('msg', '未知错误')}")
                    return None
        except Exception as e:
            if attempt < retries - 1:
                wait_time = delay * (attempt + 1)
                print(
                    f"  [{city_name}] 请求异常 (重试 {attempt + 1}/{retries}): {e}, 等待 {wait_time}秒"
                )
                time.sleep(wait_time)
            else:
                print(f"  [{city_name}] 请求失败: {e}")
                return None
    return None


def save_data(data_dir: str, start_date: str, end_date: str):
    os.makedirs(data_dir, exist_ok=True)

    total_provinces = len(PROVINCES_CITIES)

    for idx, (province, cities) in enumerate(PROVINCES_CITIES.items(), 1):
        print(f"\n[{idx}/{total_provinces}] 正在获取 {province} 省数据...")

        province_data = {}
        for city in cities:
            print(f"  正在获取 {city} 市数据...", end=" ", flush=True)
            city_data = fetch_city_data(city, start_date, end_date)

            if city_data:
                province_data[city] = city_data
                print(f"成功 ({len(city_data.get('aqi', {}).get('data', []))}条)")
            else:
                print("失败")

            time.sleep(0.5)

        file_path = os.path.join(data_dir, f"{province}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(province_data, f, ensure_ascii=False, indent=2)
        print(f"  {province} 省数据已保存: {file_path}")

        time.sleep(1)

    print(f"\n全部完成！数据保存在: {data_dir}")


if __name__ == "__main__":
    DATA_DIR = "data"
    START_DATE = "2025-10-01"
    END_DATE = "2026-03-31"

    save_data(DATA_DIR, START_DATE, END_DATE)
