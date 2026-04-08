-- 创建城市表
CREATE TABLE IF NOT EXISTS cities (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    province VARCHAR(50) NOT NULL,
    aqi INT,
    pm25 FLOAT,
    pm10 FLOAT,
    so2 FLOAT,
    no2 FLOAT,
    co FLOAT,
    o3 FLOAT,
    level VARCHAR(20),
    update_time DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 创建城市24小时数据表
CREATE TABLE IF NOT EXISTS cities_24h (
    id INT AUTO_INCREMENT PRIMARY KEY,
    city_name VARCHAR(50) NOT NULL,
    time DATETIME NOT NULL,
    aqi INT,
    pm25 FLOAT,
    pm10 FLOAT,
    so2 FLOAT,
    no2 FLOAT,
    co FLOAT,
    o3 FLOAT,
    level VARCHAR(20),
    UNIQUE KEY unique_city_time (city_name, time)
);

-- 创建城市历史数据表
CREATE TABLE IF NOT EXISTS cities_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    city_name VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    aqi INT,
    pm25 FLOAT,
    pm10 FLOAT,
    so2 FLOAT,
    no2 FLOAT,
    co FLOAT,
    o3 FLOAT,
    level VARCHAR(20),
    UNIQUE KEY unique_city_date (city_name, date)
);

-- 创建索引
CREATE INDEX idx_cities_name ON cities(name);
CREATE INDEX idx_cities_province ON cities(province);
CREATE INDEX idx_cities_24h_city_name ON cities_24h(city_name);
CREATE INDEX idx_cities_24h_time ON cities_24h(time);
CREATE INDEX idx_cities_history_city_name ON cities_history(city_name);
CREATE INDEX idx_cities_history_date ON cities_history(date);