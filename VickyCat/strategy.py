import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

DB_PATH = "market_data_async.db"

# 策略参数
VOLUME_THRESHOLD = 100  # 成交量差分阈值
TURNOVER_THRESHOLD = 10000.0  # 成交金额差分阈值
LOOKBACK_PERIOD = 10  # 秒数，用于计算高低点
EMA_SHORT_PERIOD = 9
EMA_LONG_PERIOD = 21
RSI_PERIOD = 14


def fetch_recent_kline(symbol: str, lookback: int) -> list:
    """ 获取最近 lookback 秒内的 K 线数据 """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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


def calculate_ema(prices: list, period: int) -> float:
    """ 计算 EMA 指标 """
    if len(prices) < period:
        return sum(prices) / len(prices)
    multiplier = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calculate_rsi(prices: list, period: int) -> float:
    """ 计算 RSI 指标 """
    if len(prices) < period:
        return 50  # 数据不足时返回中性值

    gains = [max(prices[i] - prices[i - 1], 0) for i in range(1, len(prices))]
    losses = [max(prices[i - 1] - prices[i], 0) for i in range(1, len(prices))]

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


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

    return None


def analyze_market_conditions(symbol: str) -> Optional[str]:
    """ 基于成交量差分、EMA、RSI 和 K 线形态进行市场分析 """
    data = fetch_recent_kline(symbol, LOOKBACK_PERIOD)

    if not data or len(data) < 2:
        return None  # 数据不足，不执行策略

    # 价格序列
    prices = [entry[4] for entry in data]

    # 计算 EMA 和 RSI
    ema_short = calculate_ema(prices, EMA_SHORT_PERIOD)
    ema_long = calculate_ema(prices, EMA_LONG_PERIOD)
    rsi = calculate_rsi(prices, RSI_PERIOD)

    # 获取最近一根 K 线
    latest_kline = data[-1]
    current_close = latest_kline[4]
    current_volume_diff = latest_kline[5]
    current_turnover_diff = latest_kline[6]

    # 识别 K 线形态
    pattern = identify_candlestick_patterns(data)

    # 信号过滤逻辑
    if pattern == "Hammer" and current_close > ema_short and rsi < 30:
        return "BUY"
    elif pattern == "Hanging Man" and current_close < ema_short and rsi > 70:
        return "SELL"
    elif ema_short > ema_long and rsi < 30:
        return "BUY"
    elif ema_short < ema_long and rsi > 70:
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
