from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests
import json
import os
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
        # 动态获取今天之前的数据（最近30个日历日）
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = today - timedelta(days=1)  # 昨天
        start_date = end_date - timedelta(days=days - 1)  # 30天前

        # 确保城市名格式正确，添加"市"后缀
        if not city_name.endswith("市") and not city_name.endswith("地区") and not city_name.endswith("自治区"):
            city_name_with_suffix = city_name + "市"
        else:
            city_name_with_suffix = city_name

        def fetch_data(sd, ed):
            """获取指定日期范围的数据"""
            params = {
                "cityname": city_name_with_suffix,
                "type": "daily",
                "start": sd.strftime("%Y-%m-%d"),
                "end": ed.strftime("%Y-%m-%d"),
            }
            print(f"请求参数: {params}")

            response = requests.get(
                "http://152.136.239.96:3000/api/history/batch-air-quality",
                headers=HEADERS,
                params=params,
                timeout=60,
            )
            data = response.json()
            if data and data.get("code") == 200:
                indicators_data = data.get("data", {}).get("indicatorsData", {})

                # 解析数据
                date_data = {}
                if isinstance(indicators_data, dict):
                    for indicator in INDICATORS:
                        indicator_obj = indicators_data.get(indicator, {})
                        if isinstance(indicator_obj, dict):
                            data_list = indicator_obj.get("data", [])
                            for item in data_list:
                                date = item.get("date")
                                value = item.get("value")
                                if date and date not in date_data:
                                    date_data[date] = {"date": date}
                                if date:
                                    date_data[date][indicator] = value
                elif isinstance(indicators_data, list):
                    for item in indicators_data:
                        if isinstance(item, dict):
                            date = item.get("date")
                            if date and date not in date_data:
                                date_data[date] = {"date": date}
                            for indicator in INDICATORS:
                                if indicator in item:
                                    date_data[date][indicator] = item[indicator]

                sorted_dates = sorted(date_data.keys())
                return [date_data[date] for date in sorted_dates]
            return None

        # 重试机制：确保获取到恰好30天的数据
        current_start = start_date
        max_retries = 2
        days_data = None

        for retry in range(max_retries + 1):
            print(f"数据获取尝试 {retry + 1}: {current_start.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
            days_data = fetch_data(current_start, end_date)

            if days_data is None:
                return None, "API返回数据失败"

            actual_days = len(days_data)
            print(f"获取到 {actual_days} 天数据")

            if actual_days == days:
                print(f"数据量符合要求，{actual_days} 天")
                break
            elif actual_days < days:
                # 数据不足，往前移一天
                print(f"数据不足({actual_days}天)，往前移一天重试")
                current_start = current_start - timedelta(days=1)
            else:
                # 数据过多，往后移一天
                print(f"数据过多({actual_days}天)，往后移一天重试")
                current_start = current_start + timedelta(days=1)

            time.sleep(1)

        if days_data is None or len(days_data) < days:
            return None, f"数据不足，仅获取到{len(days_data) if days_data else 0}天数据"

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

app.mount("/static", StaticFiles(directory="."), name="static")


def initial_sync():
    """初始化城市实时数据缓存"""
    global CITY_CACHE
    print("=" * 60)
    print("初始化城市实时数据缓存...")
    print("=" * 60)

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
        print("初始化完成，页面可正常访问")
    except Exception as e:
        print(f"初始化缓存失败: {e}")
        print("页面仍可正常访问，将使用实时数据")


@app.on_event("startup")
async def startup_event():
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, initial_sync)


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


@app.get("/api/cities/history")
async def get_city_history(cityname: str, start: str = None, end: str = None):
    """获取城市历史数据（直接从API获取，不保存到数据库）"""
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

            # 解析数据
            records = []
            date_data = {}
            for indicator in ["aqi", "pm25", "pm10", "so2", "no2", "co", "o3"]:
                indicator_obj = indicators_data.get(indicator, {})
                data_list = indicator_obj.get("data", [])
                for item in data_list:
                    date = item.get("date")
                    value = item.get("value")
                    if date and date not in date_data:
                        date_data[date] = {"date": date}
                    if date:
                        date_data[date][indicator] = value

            records = list(date_data.values())

            if records:
                filtered_records = [r for r in records if start <= r["date"] <= end]
                if filtered_records:
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
        
        # 预测日期为今天
        prediction_date = datetime.now().strftime("%Y-%m-%d")

        # 构建结果
        result = {
            "cityname": cityname,
            "province": province,
            "prediction_date": prediction_date,
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
