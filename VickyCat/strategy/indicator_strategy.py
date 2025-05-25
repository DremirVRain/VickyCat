from base_strategy import BaseStrategy
from strategy_signal import Signal, SignalType
from typing import Optional, List
from datetime import datetime


def compute_ma(data: List[float], window: int) -> float:
    return sum(data[-window:]) / window


def compute_std(data: List[float], window: int, ma: float) -> float:
    return (sum((p - ma) ** 2 for p in data[-window:]) / window) ** 0.5


def compute_rsi(prices: List[float], period: int = 14) -> float:
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
        return sum(values) / len(values)
    k = 2 / (period + 1)
    ema = values[-period]
    for price in values[-period + 1:]:
        ema = price * k + ema * (1 - k)
    return ema


def compute_macd(prices: List[float], short_period=12, long_period=26, signal_period=9):
    short_ema = compute_ema(prices, short_period)
    long_ema = compute_ema(prices, long_period)
    macd = short_ema - long_ema
    signal = compute_ema(prices[-signal_period:], signal_period)
    hist = macd - signal
    return macd, signal, hist


def simple_moving_average(prices: List[float], window: int) -> float:
    if len(prices) < window:
        return 0.0
    return sum(prices[-window:]) / window


def exponential_moving_average(prices: List[float], window: int, prev_ema: Optional[float] = None) -> float:
    if len(prices) < window:
        return 0.0
    k = 2 / (window + 1)
    if prev_ema is None:
        return simple_moving_average(prices, window)
    else:
        return prices[-1] * k + prev_ema * (1 - k)

def efficiency_ratio(prices: list[float], period: int = 10) -> float:
    """
    计算 Efficiency Ratio（效率比率），衡量趋势强度 vs 噪音。

    参数:
    - prices: float 数值列表，按时间升序排列的价格序列
    - period: 用于计算的周期长度（默认10）

    返回:
    - ER 值，范围 0~1，接近 1 趋势强，接近 0 表示震荡
    """
    if len(prices) < period + 1:
        return 0.0  # 数据不足
    
    change = abs(prices[-1] - prices[-period - 1])
    volatility = sum(abs(prices[i] - prices[i - 1]) for i in range(-period, 0))

    return change / volatility if volatility != 0 else 0.0

def calc_rate_of_change(data: List[float], period: int = 1) -> List[Optional[float]]:
    """
    计算序列的变化率 (Rate of Change, ROC)，
    ROC_t = (data_t - data_{t-period}) / data_{t-period}
    
    参数:
        data: 数值序列（如成交量、成交金额）
        period: 计算间隔周期数，默认为1
    
    返回:
        变化率列表，前 period 个元素为 None（因无数据计算）
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
    计算平滑后的变化率，先计算ROC后做简单移动平均（SMA）。
    
    参数:
        data: 数值序列
        period: 变化率计算周期
        smooth_window: SMA 平滑窗口大小
    
    返回:
        平滑后的变化率列表
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


class IndicatorStrategy(BaseStrategy):
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.cache = {}

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        raise NotImplementedError

    def is_pattern(self, *args, **kwargs) -> bool:
        raise NotImplementedError

    def build_signal(self, kline: dict, signal_type: SignalType, metadata: dict = None) -> Signal:
        return Signal(
            symbol=self.symbol,
            timestamp=datetime.strptime(kline["timestamp"], "%Y-%m-%d %H:%M:%S"),
            signal_type=signal_type,
            strength=1.0,
            strategy_name=self.__class__.__name__,
            metadata=metadata or {}
        )

    def get_params(self) -> dict:
        return {}

    def set_params(self, **params):
        pass

# ========================
# 均值回归
# ========================
class BollingerBandStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, window: int = 20, k: float = 2.0):
        super().__init__(symbol)
        self.window = window
        self.k = k
        self.prices = []

    def is_pattern(self, close: float, ma: float, upper: float, lower: float) -> Optional[SignalType]:
        if close < lower:
            return SignalType.BUY
        elif close > upper:
            return SignalType.SELL
        return None

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.prices.append(kline["close"])
        if len(self.prices) < self.window:
            return None

        ma = compute_ma(self.prices, self.window)
        std = compute_std(self.prices, self.window, ma)
        upper = ma + self.k * std
        lower = ma - self.k * std
        signal_type = self.is_pattern(kline["close"], ma, upper, lower)

        if signal_type:
            return self.build_signal(kline, signal_type, {"ma": ma, "lower": lower, "upper": upper, "std": std})
        return None

    def get_params(self) -> dict:
        return {
            "window": self.window,
            "k": self.k,
        }

    def set_params(self, **params):
        self.window = params.get("window", self.window)
        self.k = params.get("k", self.k)


# ========================
# 动量类
# ========================
class RSIStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, period: int = 14, overbought: float = 70, oversold: float = 30):
        super().__init__(symbol)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.prices = []

    def is_pattern(self, rsi: float) -> Optional[SignalType]:
        if rsi < self.oversold:
            return SignalType.BUY
        elif rsi > self.overbought:
            return SignalType.SELL
        return None

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.prices.append(kline["close"])
        if len(self.prices) < self.period + 1:
            return None

        rsi = compute_rsi(self.prices, self.period)
        signal_type = self.is_pattern(rsi)

        if signal_type:
            return self.build_signal(kline, signal_type, {"rsi": rsi})
        return None

    def get_params(self) -> dict:
        return {
            "period": self.period,
            "overbought": self.overbought,
            "oversold": self.oversold,
        }

    def set_params(self, **params):
        self.period = params.get("period", self.period)
        self.overbought = params.get("overbought", self.overbought)
        self.oversold = params.get("oversold", self.oversold)


class ATRStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, window: int = 14, atr_threshold: float = 2.0):
        super().__init__(symbol)
        self.window = window
        self.atr_threshold = atr_threshold
        self.trs, self.prev_close = [], None

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        high, low, close = kline["high"], kline["low"], kline["close"]
        if self.prev_close is None:
            self.prev_close = close
            return None

        tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.trs.append(tr)
        self.prev_close = close

        if len(self.trs) < self.window:
            return None
        atr = sum(self.trs[-self.window:]) / self.window

        metadata = {"atr": atr}
        # 假设作为动量过滤器，不直接给出交易信号，仅用于辅助其他信号判断
        return Signal(symbol=self.symbol, signal_type=SignalType.FILTER, time=kline["timestamp"], metadata=metadata)

    def get_params(self) -> dict:
        return {
            "window": self.window,
            "atr_threshold": self.atr_threshold,
        }

    def set_params(self, **params):
        self.window = params.get("window", self.window)
        self.atr_threshold = params.get("atr_threshold", self.atr_threshold)


# ========================
# 多空类
# ========================
class MACDStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, short_period: int = 12, long_period: int = 26, signal_period: int = 9):
        super().__init__(symbol)
        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period
        self.prices = []

    def is_pattern(self, macd: float, signal: float, hist: float) -> Optional[SignalType]:
        if macd > signal and hist > 0:
            return SignalType.BUY
        elif macd < signal and hist < 0:
            return SignalType.SELL
        return None

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.prices.append(kline["close"])
        if len(self.prices) < max(self.short_period, self.long_period, self.signal_period):
            return None

        macd, signal, hist = compute_macd(self.prices, self.short_period, self.long_period, self.signal_period)
        signal_type = self.is_pattern(macd, signal, hist)

        if signal_type:
            return self.build_signal(kline, signal_type, {"macd": macd, "signal": signal, "hist": hist})
        return None

    def get_params(self) -> dict:
        return {
            "short_period": self.short_period,
            "long_period": self.long_period,
            "signal_period": self.signal_period,
        }

    def set_params(self, **params):
        self.short_period = params.get("short_period", self.short_period)
        self.long_period = params.get("long_period", self.long_period)
        self.signal_period = params.get("signal_period", self.signal_period)


# ========================
# 趋势类
# ========================
class SMACrossoverStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, short_window: int = 5, long_window: int = 20):
        super().__init__(symbol)
        self.short_window = short_window
        self.long_window = long_window
        self.prices: List[float] = []
        self.last_signal: Optional[SignalType] = None

    def is_pattern(self, short_ma: float, long_ma: float) -> Optional[SignalType]:
        if short_ma > long_ma and self.last_signal != SignalType.BUY:
            return SignalType.BUY
        elif short_ma < long_ma and self.last_signal != SignalType.SELL:
            return SignalType.SELL
        return None

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.prices.append(kline["close"])
        if len(self.prices) < self.long_window:
            return None

        short_ma = simple_moving_average(self.prices, self.short_window)
        long_ma = simple_moving_average(self.prices, self.long_window)
        signal_type = self.is_pattern(short_ma, long_ma)

        if signal_type:
            self.last_signal = signal_type
            return self.build_signal(kline, signal_type, {"short_ma": short_ma, "long_ma": long_ma})
        return None

    def get_params(self) -> dict:
        return {"short_window": self.short_window, "long_window": self.long_window}

    def set_params(self, **params):
        self.short_window = params.get("short_window", self.short_window)
        self.long_window = params.get("long_window", self.long_window)


class EMACrossoverStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, short_window: int = 5, long_window: int = 20):
        super().__init__(symbol)
        self.short_window = short_window
        self.long_window = long_window
        self.prices: List[float] = []
        self.prev_short_ema: Optional[float] = None
        self.prev_long_ema: Optional[float] = None
        self.last_signal: Optional[SignalType] = None

    def is_pattern(self, short_ema: float, long_ema: float) -> Optional[SignalType]:
        if short_ema > long_ema and self.last_signal != SignalType.BUY:
            return SignalType.BUY
        elif short_ema < long_ema and self.last_signal != SignalType.SELL:
            return SignalType.SELL
        return None

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.prices.append(kline["close"])
        if len(self.prices) < self.long_window:
            return None

        short_ema = exponential_moving_average(self.prices, self.short_window, self.prev_short_ema)
        long_ema = exponential_moving_average(self.prices, self.long_window, self.prev_long_ema)

        self.prev_short_ema = short_ema
        self.prev_long_ema = long_ema

        signal_type = self.is_pattern(short_ema, long_ema)

        if signal_type:
            self.last_signal = signal_type
            return self.build_signal(kline, signal_type, {"short_ema": short_ema, "long_ema": long_ema})
        return None

    def get_params(self) -> dict:
        return {"short_window": self.short_window, "long_window": self.long_window}

    def set_params(self, **params):
        self.short_window = params.get("short_window", self.short_window)
        self.long_window = params.get("long_window", self.long_window)


class ADXStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, window: int = 14, adx_threshold: float = 25.0):
        super().__init__(symbol)
        self.window = window
        self.adx_threshold = adx_threshold
        self.highs, self.lows, self.closes = [], [], []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.highs.append(kline["high"])
        self.lows.append(kline["low"])
        self.closes.append(kline["close"])
        if len(self.highs) <= self.window:
            return None

        # 简化 ADX 计算：真实场景可调用 ta-lib 或自定义实现
        adx = self.mock_adx_calc()
        metadata = {"adx": adx}
        if adx > self.adx_threshold:
            return Signal(symbol=self.symbol, signal_type=SignalType.STRONG_TREND, time=kline["timestamp"], metadata=metadata)
        return None

    def mock_adx_calc(self) -> float:
        return 20 + 10 * ((len(self.closes) % 10) / 10)  # 模拟值用于占位

    def get_params(self) -> dict:
        return {"window": self.window, "adx_threshold": self.adx_threshold}

    def set_params(self, **params):
        self.window = params.get("window", self.window)
        self.adx_threshold = params.get("adx_threshold", self.adx_threshold)


class CCIStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, window: int = 20, threshold: float = 100.0):
        super().__init__(symbol)
        self.window = window
        self.threshold = threshold
        self.typical_prices = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        tp = (kline["high"] + kline["low"] + kline["close"]) / 3
        self.typical_prices.append(tp)
        if len(self.typical_prices) < self.window:
            return None

        ma = compute_ma(self.typical_prices, self.window)
        md = compute_std(self.typical_prices, self.window, ma)
        if md == 0:
            return None
        cci = (tp - ma) / (0.015 * md)

        metadata = {"cci": cci, "ma": ma, "md": md}
        if cci < -self.threshold:
            return self.build_signal(kline, SignalType.BUY, metadata)
        elif cci > self.threshold:
            return self.build_signal(kline, SignalType.SELL, metadata)
        return None

    def get_params(self) -> dict:
        return {"window": self.window, "threshold": self.threshold}

    def set_params(self, **params):
        self.window = params.get("window", self.window)
        self.threshold = params.get("threshold", self.threshold)


class StochasticStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, k_period: int = 14, d_period: int = 3):
        super().__init__(symbol)
        self.k_period = k_period
        self.d_period = d_period
        self.highs, self.lows, self.ks = [], [], []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.highs.append(kline["high"])
        self.lows.append(kline["low"])
        if len(self.highs) < self.k_period:
            return None

        recent_high = max(self.highs[-self.k_period:])
        recent_low = min(self.lows[-self.k_period:])
        if recent_high == recent_low:
            return None

        rsv = (kline["close"] - recent_low) / (recent_high - recent_low) * 100
        k = 2/3 * self.ks[-1] + 1/3 * rsv if self.ks else rsv
        self.ks.append(k)
        d = sum(self.ks[-self.d_period:]) / min(len(self.ks), self.d_period)

        metadata = {"k": k, "d": d}
        if k < 20 and d < 20 and k > d:
            return self.build_signal(kline, SignalType.BUY, metadata)
        elif k > 80 and d > 80 and k < d:
            return self.build_signal(kline, SignalType.SELL, metadata)
        return None

    def get_params(self) -> dict:
        return {"k_period": self.k_period, "d_period": self.d_period}

    def set_params(self, **params):
        self.k_period = params.get("k_period", self.k_period)
        self.d_period = params.get("d_period", self.d_period)
