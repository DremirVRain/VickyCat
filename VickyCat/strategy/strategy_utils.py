from typing import Optional, List, Dict
import math

def is_bullish(k):
    """阳线判断"""
    return k["close"] > k["open"]

def is_bearish(k):
    """阴线判断"""
    return k["close"] < k["open"]

def body(k):
    """实体长度"""
    return abs(k["close"] - k["open"])

def candle_range(k):
    """整根K线长度（高低价差）"""
    return k["high"] - k["low"]

def upper_shadow(k):
    """上影线长度"""
    return k["high"] - max(k["close"], k["open"])

def lower_shadow(k):
    """下影线长度"""
    return min(k["close"], k["open"]) - k["low"]

def body_ratio(k):
    """实体占整根K线比例"""
    r = candle_range(k)
    return body(k) / r if r > 0 else 0

def upper_shadow_ratio(k):
    """上影线占整根K线比例"""
    r = candle_range(k)
    return upper_shadow(k) / r if r > 0 else 0

def lower_shadow_ratio(k):
    """下影线占整根K线比例"""
    r = candle_range(k)
    return lower_shadow(k) / r if r > 0 else 0

def is_doji(k, max_body_ratio=0.1):
    """十字星判断，默认实体不超过10%"""
    return body_ratio(k) <= max_body_ratio

def is_marubozu(k, min_body_ratio=0.9, max_upper_shadow_ratio=0.05, max_lower_shadow_ratio=0.05):
    """光头光脚线判断"""
    return (
        body_ratio(k) >= min_body_ratio and
        upper_shadow_ratio(k) <= max_upper_shadow_ratio and
        lower_shadow_ratio(k) <= max_lower_shadow_ratio
    )

def is_inside_bar(current_k, prev_k):
    """内包线判断，当前K线被上一根完全包裹"""
    return (
        current_k["high"] <= prev_k["high"] and
        current_k["low"] >= prev_k["low"]
    )

def gap_up(current_k, prev_k):
    """跳空高开"""
    return current_k["low"] > prev_k["high"]

def gap_down(current_k, prev_k):
    """跳空低开"""
    return current_k["high"] < prev_k["low"]

def is_engulfing(current_k, prev_k, bullish=True):
    """
    吞没形态判断
    bullish=True判断看涨吞没（阳线包阴线）
    bullish=False判断看跌吞没（阴线包阳线）
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

def detect_trend(klines: List[dict], window: int = 5) -> Optional[dict]:
    """
    趋势识别逻辑（增强版）：
    - 收盘价连续上升/下降
    - 均线斜率
    - 效率比率（ER）
    返回字典：{ direction: "up"/"down"/"sideways", strength: 0~1 }
    """
    if len(klines) < window + 1:
        return None

    closes = [k["close"] for k in klines[-(window + 1):]]

    # 连续上升/下降次数
    up_count = sum(closes[i] < closes[i + 1] for i in range(window))
    down_count = sum(closes[i] > closes[i + 1] for i in range(window))

    # 均线斜率（当前均线与前一个均线差）
    ma_current = simple_moving_average(closes[1:], window)
    ma_prev = simple_moving_average(closes[:-1], window)
    slope = ma_current - ma_prev

    # 效率比率
    er = efficiency_ratio(closes, window)

    # 趋势方向判断
    if up_count >= window - 1 and slope > 0:
        direction = "up"
        strength = min(1.0, (up_count / window) * er * (slope / closes[-2]))
    elif down_count >= window - 1 and slope < 0:
        direction = "down"
        strength = min(1.0, (down_count / window) * er * abs(slope / closes[-2]))
    else:
        direction = "sideways"
        strength = max(0.0, er * 0.5)

    return {
        "direction": direction,
        "strength": round(strength, 3)
    }

def is_uptrend(klines: List[dict], window: int = 5) -> Dict[str, Optional[float]]:
    trend = detect_trend(klines, window)
    return trend if trend["direction"] == "up" else {"direction": "none", "strength": 0.0}

def is_downtrend(klines: List[dict], window: int = 5) -> Dict[str, Optional[float]]:
    trend = detect_trend(klines, window)
    return trend if trend["direction"] == "down" else {"direction": "none", "strength": 0.0}

def compute_ma(data: List[float], window: int) -> float:
    if len(data) < window:
        return sum(data) / len(data) if data else 0.0
    return sum(data[-window:]) / window

def compute_std(data: List[float], window: int, ma: float) -> float:
    if len(data) < window:
        return 0.0
    return (sum((p - ma) ** 2 for p in data[-window:]) / window) ** 0.5

def compute_rsi(prices: List[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0  # 无法判断时中性返回
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

def compute_ema(values: List[float], period: int) -> float:
    if len(values) < period:
        return sum(values) / len(values) if values else 0.0
    k = 2 / (period + 1)
    ema = values[-period]
    for price in values[-period + 1:]:
        ema = price * k + ema * (1 - k)
    return ema

def compute_macd(prices: List[float], short_period=12, long_period=26, signal_period=9):
    if len(prices) < long_period + signal_period:
        return 0.0, 0.0, 0.0
    short_ema = compute_ema(prices, short_period)
    long_ema = compute_ema(prices, long_period)
    macd = short_ema - long_ema

    # 修正 signal 线为 MACD 值的 EMA，而非 price
    macd_series = [compute_ema(prices[:i+1], short_period) - compute_ema(prices[:i+1], long_period)
                   for i in range(long_period, len(prices))]
    signal = compute_ema(macd_series, signal_period) if len(macd_series) >= signal_period else 0.0
    hist = macd - signal
    return macd, signal, hist

def simple_moving_average(prices: List[float], window: int) -> float:
    if len(prices) < window:
        return sum(prices) / len(prices) if prices else 0.0
    return sum(prices[-window:]) / window

def exponential_moving_average(prices: List[float], window: int, prev_ema: Optional[float] = None) -> float:
    if len(prices) < window:
        return simple_moving_average(prices, window)
    k = 2 / (window + 1)
    if prev_ema is None:
        prev_ema = simple_moving_average(prices[-window:], window)
    return prices[-1] * k + prev_ema * (1 - k)

def efficiency_ratio(prices: List[float], period: int = 10) -> float:
    """
    Efficiency Ratio（效率比率），衡量趋势 vs 噪音。
    趋势越强 ER 趋近于 1，震荡时趋近于 0
    """
    if len(prices) < period + 1:
        return 0.0
    change = abs(prices[-1] - prices[-period - 1])
    volatility = sum(abs(prices[i] - prices[i - 1]) for i in range(-period, 0))
    return change / volatility if volatility != 0 else 0.0

def calc_rate_of_change(data: List[float], period: int = 1) -> List[Optional[float]]:
    """
    计算变化率 ROC = (当前值 - 前值) / 前值
    """
    roc = [None] * len(data)
    for i in range(period, len(data)):
        prev = data[i - period]
        if prev == 0:
            roc[i] = None
        else:
            roc[i] = (data[i] - prev) / prev
    return roc

def calc_smoothed_rate_of_change(data: List[float], period: int = 1, smooth_window: int = 3) -> List[Optional[float]]:
    """
    平滑后的变化率，先计算 ROC 再做 SMA
    """
    roc = calc_rate_of_change(data, period)
    smoothed = [None] * len(roc)
    for i in range(smooth_window - 1, len(roc)):
        window_vals = [v for v in roc[i - smooth_window + 1 : i + 1] if v is not None]
        if len(window_vals) == smooth_window:
            smoothed[i] = sum(window_vals) / smooth_window
        else:
            smoothed[i] = None
    return smoothed
