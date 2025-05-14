import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

DB_PATH = "market_data_async.db"

# 策略参数
VOLUME_THRESHOLD = 100  # 成交量差分阈值
TURNOVER_THRESHOLD = 10000.0  # 成交金额差分阈值
LOOKBACK_PERIOD = 10  # 秒数，用于计算高低点

def fetch_recent_kline(symbol: str, lookback: int) -> list:
    """ 获取最近 lookback 秒内的 K 线数据 """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    end_time = datetime.now()
    start_time = end_time - timedelta(seconds=lookback)

    cursor.execute('''
        SELECT timestamp, open, high, low, close, volume_diff, turnover_diff FROM kline_1s
        WHERE symbol = ? AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
    ''', (symbol, start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S")))

    data = cursor.fetchall()
    conn.close()
    return data

def identify_candlestick_patterns(kline_data: list) -> Optional[str]:
    """ 识别常见K线形态，包括锤子线、吊颈线、吞没形态及趋势持续形态 """
    if len(kline_data) < 5:
        return None

    latest = kline_data[-1]
    prev = kline_data[-2]

    open_price = latest[1]
    high_price = latest[2]
    low_price = latest[3]
    close_price = latest[4]
    body = abs(close_price - open_price)
    upper_shadow = high_price - max(open_price, close_price)
    lower_shadow = min(open_price, close_price) - low_price

    # Hammer and Hanging Man
    if body < lower_shadow * 0.5 and lower_shadow > body * 2:
        if close_price > open_price:
            return "Hammer"
        else:
            return "Hanging Man"

    # Engulfing patterns
    if prev[1] > prev[4] and open_price < close_price and close_price > prev[1] and open_price < prev[4]:
        return "Bullish Engulfing"
    elif prev[1] < prev[4] and open_price > close_price and close_price < prev[1] and open_price > prev[4]:
        return "Bearish Engulfing"

    # Rising Three Methods & Falling Three Methods
    if len(kline_data) >= 5:
        first = kline_data[-5]
        middle = kline_data[-4:-1]
        last = kline_data[-1]

        # Rising Three Methods
        if first[4] > first[1] and all(m[4] < m[1] for m in middle) and last[4] > first[4]:
            return "Rising Three Methods"

        # Falling Three Methods
        if first[4] < first[1] and all(m[4] > m[1] for m in middle) and last[4] < first[4]:
            return "Falling Three Methods"

    return None


def analyze_market_conditions(symbol: str) -> Optional[str]:
    """ 基于成交量和成交金额差分以及K线形态进行市场分析 """
    data = fetch_recent_kline(symbol, LOOKBACK_PERIOD)

    if not data or len(data) < 2:
        return None  # 数据不足，不执行策略

    # 获取最近一根 K 线
    latest_kline = data[-1]
    high_prices = [entry[2] for entry in data]
    low_prices = [entry[3] for entry in data]
    volume_diffs = [entry[5] for entry in data]
    turnover_diffs = [entry[6] for entry in data]

    # 计算最高价和最低价
    recent_high = max(high_prices)
    recent_low = min(low_prices)
    current_close = latest_kline[4]
    current_volume_diff = latest_kline[5]
    current_turnover_diff = latest_kline[6]

    # 识别K线形态
    pattern = identify_candlestick_patterns(data)
    if pattern:
        print(f"Detected pattern: {pattern}")
        if pattern in ["Hammer", "Rising Three Methods", "Bullish Engulfing"]:
            return "BUY"
        elif pattern in ["Hanging Man", "Falling Three Methods", "Bearish Engulfing"]:
            return "SELL"

    # 策略条件判断
    if current_close > recent_high and current_volume_diff > VOLUME_THRESHOLD and current_turnover_diff > TURNOVER_THRESHOLD:
        return "BUY"
    elif current_close < recent_low and current_volume_diff < -VOLUME_THRESHOLD and current_turnover_diff < -TURNOVER_THRESHOLD:
        return "SELL"
    return None


def execute_trade(action: str, symbol: str):
    """ 模拟下单操作 """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - Executing {action} signal for {symbol}")

async def process_strategy(symbol: str):
    """ 策略循环监控 """
    while True:
        action = analyze_market_conditions(symbol)
        if action:
            execute_trade(action, symbol)
        await asyncio.sleep(1)