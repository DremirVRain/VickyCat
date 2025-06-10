from typing import Optional, List, Dict
import math

# ====== K线形态基础函数 ======

def body(k: dict) -> float:
    """K线实体长度（收盘价和开盘价的差值绝对值）"""
    return abs(k["close"] - k["open"])

def is_bullish(k: dict) -> bool:
    """阳线判断"""
    return k["close"] > k["open"]

def is_bearish(k: dict) -> bool:
    """阴线判断"""
    return k["close"] < k["open"]

def midpoint(k: dict) -> float:
    """K线实体中点价格"""
    return (k["open"] + k["close"]) / 2

def candle_range(k: dict) -> float:
    """整根K线长度（最高价减最低价）"""
    return k["high"] - k["low"]

def upper_shadow(k: dict) -> float:
    """上影线长度"""
    return k["high"] - max(k["close"], k["open"])

def lower_shadow(k: dict) -> float:
    """下影线长度"""
    return min(k["close"], k["open"]) - k["low"]

def body_ratio(k: dict) -> float:
    """实体长度占整根K线的比例"""
    r = candle_range(k)
    return body(k) / r if r > 0 else 0

def upper_shadow_ratio(k: dict) -> float:
    """上影线长度占整根K线的比例"""
    r = candle_range(k)
    return upper_shadow(k) / r if r > 0 else 0

def lower_shadow_ratio(k: dict) -> float:
    """下影线长度占整根K线的比例"""
    r = candle_range(k)
    return lower_shadow(k) / r if r > 0 else 0

# ====== 特殊K线形态判断 ======

def is_doji(k: dict, max_body_ratio: float = 0.1) -> bool:
    """十字星判断：实体占比不超过max_body_ratio，默认10%"""
    return body_ratio(k) <= max_body_ratio

def is_marubozu(k: dict, min_body_ratio: float = 0.9, max_upper_shadow_ratio: float = 0.05, max_lower_shadow_ratio: float = 0.05) -> bool:
    """光头光脚线判断：实体比例大，影线很小"""
    return (
        body_ratio(k) >= min_body_ratio and
        upper_shadow_ratio(k) <= max_upper_shadow_ratio and
        lower_shadow_ratio(k) <= max_lower_shadow_ratio
    )

def is_inside_bar(current_k: dict, prev_k: dict) -> bool:
    """内包线判断：当前K线完全被上一根包裹"""
    return (
        current_k["high"] <= prev_k["high"] and
        current_k["low"] >= prev_k["low"]
    )

def gap_up(current_k: dict, prev_k: dict) -> bool:
    """跳空高开"""
    return current_k["low"] > prev_k["high"]

def gap_down(current_k: dict, prev_k: dict) -> bool:
    """跳空低开"""
    return current_k["high"] < prev_k["low"]

def is_engulfing(current_k: dict, prev_k: dict, bullish: bool = True) -> bool:
    """
    吞没形态判断
    bullish=True 判断看涨吞没（阳线包阴线）
    bullish=False 判断看跌吞没（阴线包阳线）
    """
    if bullish:
        return (
            is_bullish(current_k) and
            is_bearish(prev_k) and
            current_k["open"] < prev_k["close"] and
            current_k["close"] > prev_k["open"]
        )
    else:
        return (
            is_bearish(current_k) and
            is_bullish(prev_k) and
            current_k["open"] > prev_k["close"] and
            current_k["close"] < prev_k["open"]
        )

# ====== 趋势检测相关 ======

def detect_trend(klines: List[dict], window: int = 5) -> Optional[dict]:
    """
    趋势识别逻辑（增强版）：
    - 收盘价连续上升/下降计数
    - 均线斜率（均线差值）
    - 效率比率（ER）
    返回字典：{ direction: "up"/"down"/"sideways", strength: 0~1 }
    """
    if len(klines) < window + 1:
        return None

    closes = [k["close"] for k in klines[-(window + 1):]]

    # 连续上涨和下跌的数量
    up_count = sum(closes[i] < closes[i + 1] for i in range(window))
    down_count = sum(closes[i] > closes[i + 1] for i in range(window))

    ma_current = simple_moving_average(closes[1:], window)
    ma_prev = simple_moving_average(closes[:-1], window)
    slope = ma_current - ma_prev

    er = efficiency_ratio(closes, window)

    if up_count >= window - 1 and slope > 0:
        direction = "up"
        strength = min(1.0, (up_count / window) * er * (slope / closes[-2]))
    elif down_count >= window - 1 and slope < 0:
        direction = "down"
        strength = min(1.0, (down_count / window) * er * abs(slope / closes[-2]))
    else:
        direction = "sideways"
        strength = max(0.0, er * 0.5)

    return {"direction": direction, "strength": round(strength, 3)}

def is_uptrend(klines: List[dict], window: int = 5) -> Dict[str, Optional[float]]:
    """判断是否为上升趋势"""
    trend = detect_trend(klines, window)
    return trend if trend and trend["direction"] == "up" else {"direction": "none", "strength": 0.0}

def is_downtrend(klines: List[dict], window: int = 5) -> Dict[str, Optional[float]]:
    """判断是否为下降趋势"""
    trend = detect_trend(klines, window)
    return trend if trend and trend["direction"] == "down" else {"direction": "none", "strength": 0.0}

# ====== 均线及技术指标计算 ======

def simple_moving_average(prices: List[float], window: int) -> float:
    """简单移动均线（SMA）"""
    if len(prices) < window:
        return sum(prices) / len(prices) if prices else 0.0
    return sum(prices[-window:]) / window

def exponential_moving_average(prices: List[float], window: int, prev_ema: Optional[float] = None) -> float:
    """指数移动均线（EMA）"""
    if len(prices) < window:
        return simple_moving_average(prices, window)
    k = 2 / (window + 1)
    if prev_ema is None:
        prev_ema = simple_moving_average(prices[-window:], window)
    return prices[-1] * k + prev_ema * (1 - k)

def efficiency_ratio(prices: List[float], period: int = 10) -> float:
    """
    Efficiency Ratio（效率比率），衡量趋势的强度与噪音
    趋势强时 ER 趋近 1，震荡时趋近 0
    """
    if len(prices) < period + 1:
        return 0.0
    change = abs(prices[-1] - prices[-period - 1])
    volatility = sum(abs(prices[i] - prices[i - 1]) for i in range(-period, 0))
    return change / volatility if volatility != 0 else 0.0

def compute_rsi(prices: List[float], period: int = 14) -> float:
    """相对强弱指标 RSI 计算"""
    if len(prices) < period + 1:
        return 50.0  # 中性值
    gains, losses = [], []
    for i in range(-period, 0):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change))
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(prices: List[float], short_period=12, long_period=26, signal_period=9):
    """MACD计算，返回 (MACD线, 信号线, 柱状图)"""
    if len(prices) < long_period + signal_period:
        return 0.0, 0.0, 0.0
    short_ema = compute_ema(prices, short_period)
    long_ema = compute_ema(prices, long_period)
    macd = short_ema - long_ema

    # 计算MACD序列用于信号线EMA
    macd_series = [compute_ema(prices[:i+1], short_period) - compute_ema(prices[:i+1], long_period)
                   for i in range(long_period, len(prices))]
    signal = compute_ema(macd_series, signal_period) if len(macd_series) >= signal_period else 0.0

    histogram = macd - signal
    return macd, signal, histogram

def compute_ema(prices: List[float], period: int) -> float:
    """辅助函数：计算EMA，返回最后一个EMA值"""
    k = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = price * k + ema * (1 - k)
    return ema

# ====== 额外实用工具 ======

def normalize_price(price: float, decimals: int = 4) -> float:
    """价格归一化，保留小数位"""
    return round(price, decimals)

def calculate_profit_loss(entry_price: float, exit_price: float, position_size: float, is_long: bool) -> float:
    """计算盈亏，适用多头和空头"""
    if is_long:
        return (exit_price - entry_price) * position_size
    else:
        return (entry_price - exit_price) * position_size

def average_price(prices: List[float]) -> float:
    """价格平均值"""
    return sum(prices) / len(prices) if prices else 0.0

def compute_std(prices: List[float], window: int, ma: float) -> float:
    """计算价格的标准差"""
    if len(prices) < window:
        return 0.0
    squared_diffs = [(price - ma) ** 2 for price in prices[-window:]]
    return math.sqrt(sum(squared_diffs) / window)