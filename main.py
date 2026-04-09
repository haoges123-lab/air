from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests
import json
import os
import pymysql
from datetime import datetime, timedelta
import time
import threading
import torch
import sys

# 添加当前目录到Python路径，确保本地模块能够正确导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 添加空气质量预测目录到Python路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '空气质量预测'))
from 数据集 import AirQualityDataset, INDICATORS, FEATURE_DIM, get_time_features
from 省份空气质量模型 import AirQualityModel

# 预测模型相关
MODEL_INDICATORS = ["so2", "no2", "co", "o3"]
MODEL_PATH = "空气质量预测/weights/best_model.pt"

# 全局模型变量
model = None
dataset_scales = None
province_to_id = None
province_names = None


def load_model():
    """加载预测模型"""
    global model, dataset_scales, province_to_id, province_names
    try:
        checkpoint = torch.load(MODEL_PATH, weights_only=True)
        dataset_scales = checkpoint["scales"]
        province_to_id = checkpoint["province_to_id"]
        province_names = checkpoint["province_names"]
        
        model = AirQualityModel(
            num_provinces=len(province_names),
            seq_len=30,
            seq_input_dim=FEATURE_DIM,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to("cpu")
        model.eval()
        print(f"模型加载成功: {MODEL_PATH}")
        print(f"省份列表: {province_names}")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False


def predict(province_name, seq_data):
    """预测空气质量"""
    global model, dataset_scales, province_to_id
    if model is None:
        load_model()
    
    # 标准化省份名称，去掉"自治区"、"省"、"市"等后缀
    normalized_province = province_name.replace("自治区", "").replace("省", "").replace("市", "").strip()
    
    # 尝试使用标准化后的省份名称
    province_id = province_to_id.get(normalized_province)
    
    # 如果还是找不到，尝试使用原始省份名称
    if province_id is None:
        province_id = province_to_id.get(province_name)
    
    # 如果仍然找不到，使用第一个省份的ID作为默认值
    if province_id is None:
        print(f"未知省份: {province_name}，使用默认省份ID")
        if province_to_id:
            province_id = next(iter(province_to_id.values()))
        else:
            raise ValueError(f"未知省份: {province_name}")

    model.eval()
    with torch.inference_mode():
        province_ids = torch.tensor([province_id])
        inputs = seq_data.unsqueeze(0)
        pred = model(province_ids, inputs).squeeze()

    return pred.numpy()


def prepare_prediction_data(city_name, province_name, days=30):
    """准备预测数据"""
    try:
        # 固定数据获取范围为3月2日到4月1日（30天）
        start_date = datetime(2026, 3, 2)
        end_date = datetime(2026, 4, 1)
        print(f"数据获取范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
        print(f"请求城市: {city_name}")
        
        # 确保城市名格式正确，添加"市"后缀
        if not city_name.endswith("市") and not city_name.endswith("地区") and not city_name.endswith("自治区"):
            city_name_with_suffix = city_name + "市"
        else:
            city_name_with_suffix = city_name
        
        params = {
            "cityname": city_name_with_suffix,
            "type": "daily",
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
        }
        print(f"实际请求参数: {params}")
        
        # 添加重试机制，处理API请求频率限制
        max_retries = 3
        retry_interval = 2  # 秒
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    "http://152.136.239.96:3000/api/history/batch-air-quality",
                    headers=HEADERS,
                    params=params,
                    timeout=60,
                )
                print(f"第 {attempt + 1} 次请求，状态码: {response.status_code}")
                break
            except requests.RequestException as e:
                print(f"请求失败，尝试 {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_interval)
                else:
                    return None, f"请求失败: {str(e)}"
        
        # 添加请求间隔，避免触发API频率限制
        time.sleep(1)
        
        data = response.json()
        print(f"API响应状态码: {response.status_code}")
        print(f"API响应数据: {data}")
        if data and data.get("code") == 200:
            indicators_data = data.get("data", {}).get("indicatorsData", {})
            print(f"indicatorsData: {indicators_data}")
            
            # 解析数据
            date_data = {}
            print(f"开始解析数据，指标列表: {INDICATORS}")
            
            # 尝试不同的数据结构格式
            if isinstance(indicators_data, dict):
                # 第一种格式: indicatorsData是包含各个指标的字典
                for indicator in INDICATORS:
                    indicator_obj = indicators_data.get(indicator, {})
                    if isinstance(indicator_obj, dict):
                        data_list = indicator_obj.get("data", [])
                        print(f"指标 {indicator} 有 {len(data_list)} 条数据")
                        for item in data_list:
                            date = item.get("date")
                            value = item.get("value")
                            if date and date not in date_data:
                                date_data[date] = {"date": date}
                            if date:
                                date_data[date][indicator] = value
            elif isinstance(indicators_data, list):
                # 第二种格式: indicatorsData是列表
                print(f"indicatorsData是列表，长度: {len(indicators_data)}")
                for item in indicators_data:
                    if isinstance(item, dict):
                        date = item.get("date")
                        if date and date not in date_data:
                            date_data[date] = {"date": date}
                        for indicator in INDICATORS:
                            if indicator in item:
                                date_data[date][indicator] = item[indicator]
            
            print(f"解析完成，共获取到 {len(date_data)} 天数据")
            
            # 按日期排序
            sorted_dates = sorted(date_data.keys())
            days_data = [date_data[date] for date in sorted_dates]
            
            if len(days_data) < days:
                return None, f"数据不足，仅获取到{len(days_data)}天数据"
            
            # 准备输入数据
            scales = dataset_scales.get(province_name, {})
            # 如果找不到该省份的缩放参数，使用默认值
            if not scales:
                print(f"未找到{province_name}的缩放参数，使用默认值")
                # 使用第一个省份的缩放参数作为默认值
                if dataset_scales:
                    first_province = next(iter(dataset_scales))
                    scales = dataset_scales[first_province]
                else:
                    # 如果没有任何缩放参数，使用默认值
                    scales = {ind: (0.0, 1.0) for ind in INDICATORS}
            
            inp_vals = []
            for day in days_data:
                row = []
                for ind in INDICATORS:
                    v = day.get(ind)
                    mean, std = scales.get(ind, (0.0, 1.0))
                    row.append((v - mean) / std if v is not None else 0.0)
                time_feat = get_time_features(day["date"])
                row.extend(time_feat)
                inp_vals.append(row)
            
            return torch.tensor(inp_vals, dtype=torch.float32), None
        else:
            return None, "API返回数据失败"
    except Exception as e:
        return None, str(e)

app = FastAPI(title="中国空气质量可视化系统")

CITY_DATA_FILE = "china_cities.json"

EXCLUDE_REGIONS = ["香港特别行政区", "澳门特别行政区", "台湾省"]

try:
    with open(CITY_DATA_FILE, "r", encoding="utf-8") as f:
        CITIES_DATA = json.load(f)
    CITIES_DATA = {k: v for k, v in CITIES_DATA.items() if k not in EXCLUDE_REGIONS}
except Exception as e:
        print(f"加载城市数据失败: {e}")
        CITIES_DATA = {}

CITY_TO_PROVINCE = {}
CITY_CACHE = {}
CITY_CACHE_LOCK = threading.Lock()

CACHE_24H = {}
CACHE_24H_LOCK = threading.Lock()
CACHE_EXPIRY = 300

VALID_CITIES = set()

for province, cities in CITIES_DATA.items():
    for city in cities:
        normalized = city.replace("市", "").replace("地区", "")
        CITY_TO_PROVINCE[city] = province
        CITY_TO_PROVINCE[normalized] = province
        VALID_CITIES.add(city)
        VALID_CITIES.add(normalized)
        if not city.endswith("市") and not city.endswith("地区"):
            VALID_CITIES.add(city + "市")

API_URL = "http://152.136.239.96:3000/api/latest-city-data"
API_24H_URL = "http://152.136.239.96:3000/api/air/24h-data"
API_HISTORY_URL = "http://152.136.239.96:3000/api/history/batch-air-quality"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Referer": "http://152.136.239.96:3000/",
    "Origin": "http://152.136.239.96:3000",
    "X-Requested-With": "XMLHttpRequest",
}

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "123456"),
    "database": os.getenv("DB_NAME", "air_quality"),
    "charset": "utf8mb4",
}

app.mount("/static", StaticFiles(directory="."), name="static")


def init_db():
    try:
        conn = pymysql.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            charset="utf8mb4",
        )
        cursor = conn.cursor()
        cursor.execute(
            "CREATE DATABASE IF NOT EXISTS air_quality CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        cursor.close()
        conn.close()

        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS air_quality_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                city_name VARCHAR(50) NOT NULL,
                province VARCHAR(50) NOT NULL,
                date DATE NOT NULL,
                aqi DECIMAL(10,2),
                pm25 DECIMAL(10,2),
                pm10 DECIMAL(10,2),
                so2 DECIMAL(10,2),
                no2 DECIMAL(10,2),
                co DECIMAL(10,2),
                o3 DECIMAL(10,2),
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uk_city_date (city_name, date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS 24_hour_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                city_name VARCHAR(50) NOT NULL,
                province VARCHAR(50) NOT NULL,
                data_hour DATETIME NOT NULL,
                aqi DECIMAL(10,2),
                pm25 DECIMAL(10,2),
                pm10 DECIMAL(10,2),
                so2 DECIMAL(10,2),
                no2 DECIMAL(10,2),
                co DECIMAL(10,2),
                o3 DECIMAL(10,2),
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uk_city_hour (city_name, data_hour)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"数据库初始化失败: {e}")
        return False


def clean_exclude_regions():
    excluded = ["香港特别行政区", "澳门特别行政区", "台湾省"]
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        for region in excluded:
            cursor.execute(
                "DELETE FROM air_quality_history WHERE province = %s", (region,)
            )
            deleted = cursor.rowcount
            if deleted > 0:
                print(f"  已删除 {deleted} 条 {region} 数据")
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"清理排除区域数据失败: {e}")


def get_date_range():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=9)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def fetch_historical_from_api(city_name, start_date, end_date):
    try:
        normalized_city = city_name.replace("市", "").replace("地区", "")
        params = {
            "cityname": normalized_city,
            "type": "daily",
            "start": start_date,
            "end": end_date,
        }
        response = requests.get(
            API_HISTORY_URL, headers=HEADERS, params=params, timeout=60
        )
        return response.json()
    except Exception as e:
        print(f"获取历史数据失败: {city_name}, 错误: {e}")
        return None


def parse_indicators_data(indicators_data, city_name, province):
    records = []
    if not indicators_data:
        return records
    indicators = ["aqi", "pm25", "pm10", "so2", "no2", "co", "o3"]
    date_data = {}
    for indicator in indicators:
        indicator_obj = indicators_data.get(indicator, {})
        data_list = indicator_obj.get("data", [])
        for item in data_list:
            date = item.get("date")
            value = item.get("value")
            if date and date not in date_data:
                date_data[date] = {
                    "city_name": city_name,
                    "province": province,
                    "date": date,
                }
            if date:
                date_data[date][indicator] = value
    records = list(date_data.values())
    return records


def save_to_db(records):
    if not records:
        return 0
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        insert_sql = """
            INSERT INTO air_quality_history
            (city_name, province, date, aqi, pm25, pm10, so2, no2, co, o3)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            aqi = VALUES(aqi), pm25 = VALUES(pm25), pm10 = VALUES(pm10),
            so2 = VALUES(so2), no2 = VALUES(no2), co = VALUES(co), o3 = VALUES(o3)
        """
        count = 0
        for record in records:
            try:
                cursor.execute(
                    insert_sql,
                    (
                        record["city_name"],
                        record["province"],
                        record["date"],
                        record.get("aqi"),
                        record.get("pm25"),
                        record.get("pm10"),
                        record.get("so2"),
                        record.get("no2"),
                        record.get("co"),
                        record.get("o3"),
                    ),
                )
                count += 1
            except Exception as e:
                pass
        conn.commit()
        cursor.close()
        conn.close()
        return count
    except Exception as e:
        print(f"保存数据失败: {e}")
        return 0


def get_all_cities_with_data():
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT city_name, province, MAX(date) as last_date, COUNT(*) as data_count
            FROM air_quality_history
            GROUP BY city_name, province
        """)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        cities_data = {}
        for row in results:
            cities_data[row[0]] = {
                "province": row[1],
                "last_date": row[2],
                "data_count": row[3],
            }
        return cities_data
    except Exception as e:
        print(f"获取城市数据失败: {e}")
        return {}


def delete_old_data():
    try:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = today - timedelta(days=10)
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM air_quality_history WHERE date < %s",
            (cutoff_date.strftime("%Y-%m-%d"),),
        )
        deleted = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        if deleted > 0:
            print(f"  已删除 {deleted} 条过期数据(10天前)")
    except Exception as e:
        print(f"删除过期数据失败: {e}")


def update_history_data():
    print("\n" + "=" * 60)
    print("开始定时数据同步...")
    print(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    target_end_date = today - timedelta(days=1)
    target_start_date = target_end_date - timedelta(days=9)

    start_date_str = target_start_date.strftime("%Y-%m-%d")
    end_date_str = target_end_date.strftime("%Y-%m-%d")
    print(f"目标日期范围: {start_date_str} 至 {end_date_str}")

    delete_old_data()

    cities_with_data = get_all_cities_with_data()
    print(f"数据库已有 {len(cities_with_data)} 个城市的历史数据")

    need_update_cities = []
    for province, cities in CITIES_DATA.items():
        for city in cities:
            if city in cities_with_data:
                last_date = cities_with_data[city]["last_date"]
                if isinstance(last_date, str):
                    last_date = datetime.strptime(last_date, "%Y-%m-%d").date()
                elif hasattr(last_date, "date"):
                    last_date = last_date.date()
                target_end_date_only = target_end_date.date()
                if last_date < target_end_date_only:
                    need_update_cities.append((city, province, "数据过期"))
                elif cities_with_data[city]["data_count"] < 10:
                    need_update_cities.append((city, province, "数据不足"))
            else:
                need_update_cities.append((city, province, "无数据"))

    provinces = sorted(CITIES_DATA.keys())
    total_cities = 0
    success_cities = 0
    total_records = 0

    for province in provinces:
        cities = CITIES_DATA[province]
        province_need_update = [
            (c, p, r) for c, p, r in need_update_cities if c in cities
        ]
        if not province_need_update:
            continue
        print(f"\n正在处理: {province} ({len(province_need_update)} 个城市需更新)")
        province_records = []
        for i, (city, _, reason) in enumerate(province_need_update):
            total_cities += 1
            print(
                f"  [{i + 1}/{len(province_need_update)}] {city} ({reason})...",
                end=" ",
                flush=True,
            )
            data = fetch_historical_from_api(city, start_date_str, end_date_str)
            if data and data.get("code") == 200:
                indicators_data = data.get("data", {}).get("indicatorsData", {})
                records = parse_indicators_data(indicators_data, city, province)
                province_records.extend(records)
                success_cities += 1
                print(f"成功 ({len(records)} 条)")
            else:
                print("失败")
            time.sleep(1)
        if province_records:
            saved = save_to_db(province_records)
            total_records += saved

    print("\n" + "=" * 60)
    print(
        f"定时同步完成! 检查: {total_cities}, 成功: {success_cities}, 新增记录: {total_records}"
    )
    print("=" * 60)


def save_24h_data(records):
    """保存24小时数据到数据库"""
    if not records:
        return 0
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        insert_sql = """
            INSERT INTO 24_hour_data
            (city_name, province, data_hour, aqi, pm25, pm10, so2, no2, co, o3)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            aqi = VALUES(aqi), pm25 = VALUES(pm25), pm10 = VALUES(pm10),
            so2 = VALUES(so2), no2 = VALUES(no2), co = VALUES(co), o3 = VALUES(o3)
        """
        count = 0
        for record in records:
            try:
                cursor.execute(
                    insert_sql,
                    (
                        record["city_name"],
                        record["province"],
                        record["data_hour"],
                        record.get("aqi"),
                        record.get("pm25"),
                        record.get("pm10"),
                        record.get("so2"),
                        record.get("no2"),
                        record.get("co"),
                        record.get("o3"),
                    ),
                )
                count += 1
            except Exception as e:
                print(f"  插入失败: {record.get('city_name')}, {record.get('data_hour')}: {e}")
        conn.commit()
        cursor.close()
        conn.close()
        return count
    except Exception as e:
        print(f"保存24小时数据失败: {e}")
        return 0


def update_24h_data():
    """更新所有城市的24小时数据"""
    print("=" * 60)
    print("开始更新24小时数据...")
    print("=" * 60)

    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT city_name FROM 24_hour_data")
    existing_cities = set(row[0] for row in cursor.fetchall())
    cursor.close()
    conn.close()

    cities_to_fetch = list(VALID_CITIES)
    total = len(cities_to_fetch)
    success = 0
    total_records = 0

    for i, city in enumerate(cities_to_fetch):
        print(f"[{i + 1}/{total}] 获取 {city} ...", end=" ", flush=True)

        city_names_to_try = [city, city + "市"]
        all_data = None

        for cn in city_names_to_try:
            try:
                response = requests.get(
                    f"http://152.136.239.96:3000/api/air/batch-24h-data?type=city&cityname={cn}",
                    headers=HEADERS,
                    timeout=30,
                )
                result = response.json()
                if result.get("code") != 200:
                    continue

                data = result.get("data", {})
                indicators_data = data.get("indicators", {})
                common_hours = data.get("commonHours", [])

                aqi_values = indicators_data.get("aqi", [])
                if not aqi_values or all(v == 0 or v is None for v in aqi_values):
                    continue

                indicator_names = ["aqi", "pm25", "pm10", "so2", "no2", "co", "o3"]
                records = []
                province = CITY_TO_PROVINCE.get(city, "未知")

                for j, hour_str in enumerate(common_hours):
                    try:
                        dt = datetime.strptime(hour_str, "%Y-%m-%d %H")
                        record = {
                            "city_name": city,
                            "province": province,
                            "data_hour": dt,
                        }
                        for ind in indicator_names:
                            values = indicators_data.get(ind, [])
                            record[ind] = values[j] if j < len(values) else None
                        records.append(record)
                    except Exception:
                        continue

                if records:
                    saved = save_24h_data(records)
                    total_records += saved
                    all_data = records
                    break
            except Exception as e:
                continue

        if all_data:
            success += 1
            print(f"成功 ({len(all_data)} 条)")
        else:
            print("失败")

        time.sleep(0.5)

    print("=" * 60)
    print(
        f"24小时数据更新完成! 总数: {total}, 成功: {success}, 新增记录: {total_records}"
    )
    print("=" * 60)


def background_24h_update_task():
    """后台24小时数据更新任务，每30分钟执行一次"""
    while True:
        try:
            update_24h_data()
        except Exception as e:
            print(f"24小时数据更新异常: {e}")
        time.sleep(1800)


def background_update_task():
    while True:
        time.sleep(300)
        update_history_data()


def initial_sync():
    global CITY_CACHE
    print("=" * 60)
    print("启动数据同步服务...")
    print("=" * 60)

    print("初始化城市实时数据缓存...")
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=30)
        result = response.json()
        data = result.get("data", [])
        with CITY_CACHE_LOCK:
            for item in data:
                city_name = item.get("city", "") or item.get("cityname", "")
                normalized = city_name.replace("市", "").replace("地区", "")
                province = (
                    CITY_TO_PROVINCE.get(city_name)
                    or CITY_TO_PROVINCE.get(normalized)
                    or "未知"
                )
                item["province"] = province
                item["cityname"] = city_name
                CITY_CACHE[city_name] = item
                CITY_CACHE[normalized] = item
        print(f"已缓存 {len(CITY_CACHE)} 个城市数据")
    except Exception as e:
        print(f"初始化缓存失败: {e}")

    if not init_db():
        print("数据库初始化失败，跳过历史数据同步")
        return

    print("清理排除区域数据...")
    clean_exclude_regions()

    start_date, end_date = get_date_range()
    print(f"获取日期范围: {start_date} 至 {end_date} (共10天)")

    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM air_quality_history")
    existing_count = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    if existing_count > 0:
        print(f"数据库已有 {existing_count} 条历史数据，进行增量更新...")
        update_history_data()
    else:
        print("开始首次同步历史数据...")
        provinces = sorted(CITIES_DATA.keys())
        total_cities = 0
        success_cities = 0
        total_records = 0

        for province in provinces:
            cities = CITIES_DATA[province]
            print(f"\n正在处理: {province} ({len(cities)} 个城市)")
            province_records = []

            for i, city in enumerate(cities):
                total_cities += 1
                print(f"  [{i + 1}/{len(cities)}] {city}...", end=" ", flush=True)
                data = fetch_historical_from_api(city, start_date, end_date)
                if data and data.get("code") == 200:
                    indicators_data = data.get("data", {}).get("indicatorsData", {})
                    records = parse_indicators_data(indicators_data, city, province)
                    province_records.extend(records)
                    success_cities += 1
                    print(f"成功 ({len(records)} 条)")
                else:
                    print("失败")
                time.sleep(1)

            if province_records:
                saved = save_to_db(province_records)
                total_records += saved

        print("\n" + "=" * 60)
        print(
            f"历史数据同步完成! 成功: {success_cities}/{total_cities}, 记录: {total_records}"
        )
        print("=" * 60)

    update_thread = threading.Thread(target=background_update_task, daemon=True)
    update_thread.start()
    print("历史数据定时更新线程已启动(每5分钟)...")

    print("开始首次获取24小时数据...")
    update_24h_data()
    h24_thread = threading.Thread(target=background_24h_update_task, daemon=True)
    h24_thread.start()
    print("24小时数据定时更新线程已启动(每30分钟)...")


@app.on_event("startup")
async def startup_event():
    import asyncio

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, initial_sync)
    print("后台数据同步已启动，页面可正常访问...")


@app.get("/")
async def root():
    return FileResponse("index.html")


def load_cities_from_json():
    """从本地JSON文件加载城市数据，只返回china_cities.json中存在的城市"""
    try:
        if os.path.exists("city_air_data.json"):
            with open("city_air_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                filtered_data = []
                for item in data:
                    city_name = item.get("cityname", "") or item.get("city", "")
                    normalized = city_name.replace("市", "").replace("地区", "")
                    if city_name in VALID_CITIES or normalized in VALID_CITIES:
                        filtered_data.append(item)
                return filtered_data
    except Exception as e:
        print(f"读取本地JSON失败: {e}")
    return None


@app.get("/api/cities")
async def get_cities(save: bool = False):
    global CITY_CACHE

    if not save:
        local_data = load_cities_from_json()
        if local_data:
            with CITY_CACHE_LOCK:
                CITY_CACHE.clear()
                for item in local_data:
                    city_name = item.get("cityname", "") or item.get("city", "")
                    if city_name:
                        CITY_CACHE[city_name] = item
                        normalized = city_name.replace("市", "").replace("地区", "")
                        CITY_CACHE[normalized] = item
            print(f"从本地JSON加载 {len(local_data)} 个城市数据")
            return {
                "code": 200,
                "data": local_data,
                "count": len(local_data),
                "source": "本地文件",
            }

    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=30)
        result = response.json()
        data = result.get("data", [])

        with CITY_CACHE_LOCK:
            CITY_CACHE.clear()
            filtered_data = []
            for item in data:
                city_name = item.get("city", "") or item.get("cityname", "")
                normalized = city_name.replace("市", "").replace("地区", "")
                if city_name not in VALID_CITIES and normalized not in VALID_CITIES:
                    continue
                province = (
                    CITY_TO_PROVINCE.get(city_name)
                    or CITY_TO_PROVINCE.get(normalized)
                    or "未知"
                )
                item["province"] = province
                item["cityname"] = city_name
                CITY_CACHE[city_name] = item
                CITY_CACHE[normalized] = item
                filtered_data.append(item)
        processed_data = filtered_data

        with open("city_air_data.json", "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        return {"code": 200, "data": processed_data, "count": len(processed_data)}
    except Exception as e:
        with CITY_CACHE_LOCK:
            processed_data = list(CITY_CACHE.values())
        if processed_data:
            return {
                "code": 200,
                "data": processed_data,
                "count": len(processed_data),
                "warning": "使用缓存数据",
            }
        return {"code": 500, "msg": str(e), "data": []}


@app.get("/api/cities/24h")
async def get_city_24h(cityname: str):
    try:
        original_name = cityname
        city_name = cityname.replace("市", "").replace("地区", "")

        with CACHE_24H_LOCK:
            if city_name in CACHE_24H:
                cached = CACHE_24H[city_name]
                if time.time() - cached["timestamp"] < CACHE_EXPIRY:
                    return {"code": 200, "data": cached["data"], "source": "缓存"}

        city_names_to_try = [city_name, original_name, original_name + "市"]
        all_data = None

        for cn in city_names_to_try:
            response = requests.get(
                f"http://152.136.239.96:3000/api/air/batch-24h-data?type=city&cityname={cn}",
                headers=HEADERS,
                timeout=30,
            )
            result = response.json()

            if result.get("code") != 200:
                continue

            data = result.get("data", {})
            indicators_data = data.get("indicators", {})
            common_hours = data.get("commonHours", [])

            aqi_values = indicators_data.get("aqi", [])
            if not aqi_values or all(v == 0 for v in aqi_values):
                continue

            indicator_names = ["aqi", "pm25", "pm10", "so2", "no2", "co", "o3"]
            all_data = {}

            for ind in indicator_names:
                values = indicators_data.get(ind, [])
                merged = []
                for i, hour in enumerate(common_hours):
                    time_str = hour.split(" ")[1] if " " in hour else hour
                    merged.append(
                        {
                            "time": time_str,
                            "value": values[i] if i < len(values) else None,
                        }
                    )
                all_data[ind] = merged
            break

        if all_data is None:
            return {"code": 404, "msg": "暂无24小时数据", "data": {}}

        with CACHE_24H_LOCK:
            CACHE_24H[city_name] = {"data": all_data, "timestamp": time.time()}

        return {"code": 200, "data": all_data, "source": "实时获取"}
    except Exception as e:
        with CACHE_24H_LOCK:
            if city_name in CACHE_24H:
                return {
                    "code": 200,
                    "data": CACHE_24H[city_name]["data"],
                    "source": "缓存(过期)",
                }
        return {"code": 500, "msg": str(e), "data": {}}


@app.get("/api/cities/24h/db")
async def get_city_24h_from_db(cityname: str):
    """从数据库获取24小时数据"""
    try:
        city_name = cityname.replace("市", "").replace("地区", "")

        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT city_name, province, data_hour, aqi, pm25, pm10, so2, no2, co, o3 
               FROM 24_hour_data 
               WHERE city_name = %s OR city_name = %s
               ORDER BY data_hour DESC
               LIMIT 100""",
            (cityname, city_name),
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if rows:
            records = []
            for row in rows:
                records.append(
                    {
                        "city_name": row[0],
                        "province": row[1],
                        "time": row[2].strftime("%Y-%m-%d %H:00"),
                        "hour": row[2].strftime("%H"),
                        "aqi": float(row[3]) if row[3] else None,
                        "pm25": float(row[4]) if row[4] else None,
                        "pm10": float(row[5]) if row[5] else None,
                        "so2": float(row[6]) if row[6] else None,
                        "no2": float(row[7]) if row[7] else None,
                        "co": float(row[8]) if row[8] else None,
                        "o3": float(row[9]) if row[9] else None,
                    }
                )
            return {
                "code": 200,
                "data": records,
                "source": "数据库",
                "count": len(records),
            }

        return {"code": 404, "msg": "数据库中暂无该城市数据", "data": [], "count": 0}
    except Exception as e:
        return {"code": 500, "msg": str(e), "data": []}


@app.get("/api/cities/history")
async def get_city_history(cityname: str, start: str = None, end: str = None):
    try:
        city_name = cityname

        if not start or not end:
            end_date = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=1)
            start_date = end_date - timedelta(days=9)
            start = start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d")

        params = {
            "cityname": city_name,
            "type": "daily",
            "start": start,
            "end": end,
        }
        response = requests.get(
            "http://152.136.239.96:3000/api/history/batch-air-quality",
            headers=HEADERS,
            params=params,
            timeout=60,
        )
        data = response.json()

        if data and data.get("code") == 200:
            indicators_data = data.get("data", {}).get("indicatorsData", {})
            province = (
                CITY_TO_PROVINCE.get(cityname)
                or CITY_TO_PROVINCE.get(cityname.replace("市", "").replace("地区", ""))
                or "未知"
            )
            records = parse_indicators_data(indicators_data, cityname, province)

            if records:
                filtered_records = [r for r in records if start <= r["date"] <= end]
                if filtered_records:
                    threading.Thread(
                        target=save_to_db, args=(filtered_records,), daemon=True
                    ).start()
                    return {
                        "code": 200,
                        "source": "官网获取",
                        "start": start,
                        "end": end,
                        "data": filtered_records,
                        "url": "http://eairmap.cn/",
                    }

        return {
            "code": 404,
            "msg": "官网未返回数据，可能是城市名称不匹配",
            "start": start,
            "end": end,
            "data": [],
            "url": "http://eairmap.cn/",
        }

    except Exception as e:
        print(f"历史数据查询异常: {cityname}, 错误: {e}")
        return {"code": 500, "msg": f"服务器错误: {str(e)}", "data": []}


@app.get("/api/provinces")
async def get_provinces():
    provinces_data = {}
    for province, cities in CITIES_DATA.items():
        provinces_data[province] = cities
    return {"code": 200, "data": provinces_data}


@app.get("/china_map.json")
async def get_china_map():
    return FileResponse("china_map.json", media_type="application/json")


@app.get("/api/cities/predict")
async def predict_city_air_quality(cityname: str):
    """预测城市空气质量"""
    try:
        # 加载模型
        if model is None:
            load_model()
        
        # 获取城市对应的省份
        city_name = cityname.replace("市", "").replace("地区", "")
        province = CITY_TO_PROVINCE.get(cityname) or CITY_TO_PROVINCE.get(city_name) or "未知"
        
        if province == "未知":
            return {"code": 404, "msg": "未找到城市对应的省份", "data": {}}
        
        # 准备预测数据
        seq_data, error = prepare_prediction_data(cityname, province)
        if seq_data is None:
            return {"code": 400, "msg": error, "data": {}}
        
        # 预测
        pred = predict(province, seq_data)
        
        # 反缩放
        scales = dataset_scales.get(province, {})
        # 如果找不到该省份的缩放参数，使用默认值
        if not scales:
            print(f"未找到{province}的缩放参数，使用默认值进行反缩放")
            # 使用第一个省份的缩放参数作为默认值
            if dataset_scales:
                first_province = next(iter(dataset_scales))
                scales = dataset_scales[first_province]
            else:
                # 如果没有任何缩放参数，使用默认值
                scales = {ind: (0.0, 1.0) for ind in INDICATORS}
        
        pred_unscaled = []
        for j, ind in enumerate(INDICATORS):
            mean, std = scales.get(ind, (0.0, 1.0))
            pred_unscaled.append(pred[j] * std + mean)
        
        # 构建结果
        result = {
            "cityname": cityname,
            "province": province,
            "prediction_date": "2026-04-08",
            "indicators": {}
        }
        
        for i, ind in enumerate(INDICATORS):
            # 确保将numpy类型转换为Python原生类型
            result["indicators"][ind] = round(float(pred_unscaled[i]), 2)
        
        return {
            "code": 200,
            "data": result,
            "msg": "预测成功"
        }
    except Exception as e:
        return {
            "code": 500,
            "msg": f"预测失败: {str(e)}",
            "data": {}
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
