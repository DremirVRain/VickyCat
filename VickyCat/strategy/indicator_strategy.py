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
    for price in values[-period+1:]:
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
        # 初始EMA为同样长度的简单均线
        return simple_moving_average(prices, window)
    else:
        return prices[-1] * k + prev_ema * (1 - k)

class IndicatorStrategy(BaseStrategy):
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.cache = {}

    def generate_signal(self, kline: dict) -> Optional[Signal]:
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


class BollingerBandStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, window: int = 20, k: float = 2.0):
        super().__init__(symbol)
        self.window = window
        self.k = k
        self.prices = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.prices.append(kline["close"])
        if len(self.prices) < self.window:
            return None

        ma = compute_ma(self.prices, self.window)
        std = compute_std(self.prices, self.window, ma)
        upper = ma + self.k * std
        lower = ma - self.k * std

        if kline["close"] < lower:
            return self.build_signal(kline, SignalType.BUY, {"ma": ma, "lower": lower, "upper": upper, "std": std})
        elif kline["close"] > upper:
            return self.build_signal(kline, SignalType.SELL, {"ma": ma, "lower": lower, "upper": upper, "std": std})

        return None


class RSIStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, period: int = 14, overbought: float = 70, oversold: float = 30):
        super().__init__(symbol)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.prices = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.prices.append(kline["close"])
        if len(self.prices) < self.period + 1:
            return None

        rsi = compute_rsi(self.prices, self.period)

        if rsi < self.oversold:
            return self.build_signal(kline, SignalType.BUY, {"rsi": rsi})
        elif rsi > self.overbought:
            return self.build_signal(kline, SignalType.SELL, {"rsi": rsi})

        return None


class MACDStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, short_period: int = 12, long_period: int = 26, signal_period: int = 9):
        super().__init__(symbol)
        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period
        self.prices = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.prices.append(kline["close"])
        if len(self.prices) < max(self.short_period, self.long_period, self.signal_period):
            return None

        macd, signal, hist = compute_macd(self.prices, self.short_period, self.long_period, self.signal_period)

        if macd > signal and hist > 0:
            return self.build_signal(kline, SignalType.BUY, {"macd": macd, "signal": signal, "hist": hist})
        elif macd < signal and hist < 0:
            return self.build_signal(kline, SignalType.SELL, {"macd": macd, "signal": signal, "hist": hist})

        return None

class SMACrossoverStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, short_window: int = 5, long_window: int = 20):
        super().__init__(symbol)
        self.short_window = short_window
        self.long_window = long_window
        self.prices: List[float] = []
        self.last_signal: Optional[SignalType] = None

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.prices.append(kline["close"])
        if len(self.prices) < self.long_window:
            return None

        short_ma = simple_moving_average(self.prices, self.short_window)
        long_ma = simple_moving_average(self.prices, self.long_window)

        metadata = {"short_ma": short_ma, "long_ma": long_ma}

        # 判断金叉死叉
        if short_ma > long_ma:
            if self.last_signal != SignalType.BUY:
                self.last_signal = SignalType.BUY
                return self.build_signal(kline, SignalType.BUY, metadata)
        elif short_ma < long_ma:
            if self.last_signal != SignalType.SELL:
                self.last_signal = SignalType.SELL
                return self.build_signal(kline, SignalType.SELL, metadata)
        return None


class EMACrossoverStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, short_window: int = 5, long_window: int = 20):
        super().__init__(symbol)
        self.short_window = short_window
        self.long_window = long_window
        self.prices: List[float] = []
        self.prev_short_ema: Optional[float] = None
        self.prev_long_ema: Optional[float] = None
        self.last_signal: Optional[SignalType] = None

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.prices.append(kline["close"])
        if len(self.prices) < self.long_window:
            return None

        short_ema = exponential_moving_average(self.prices, self.short_window, self.prev_short_ema)
        long_ema = exponential_moving_average(self.prices, self.long_window, self.prev_long_ema)

        self.prev_short_ema = short_ema
        self.prev_long_ema = long_ema

        metadata = {"short_ema": short_ema, "long_ema": long_ema}

        # 判断金叉死叉
        if short_ema > long_ema:
            if self.last_signal != SignalType.BUY:
                self.last_signal = SignalType.BUY
                return self.build_signal(kline, SignalType.BUY, metadata)
        elif short_ema < long_ema:
            if self.last_signal != SignalType.SELL:
                self.last_signal = SignalType.SELL
                return self.build_signal(kline, SignalType.SELL, metadata)
        return None