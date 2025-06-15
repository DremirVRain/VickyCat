from base_strategy import BaseStrategy, MarketContext
from strategy_signal import Signal, SignalType
from typing import Optional, List
from datetime import datetime
from strategy.strategy_utils import *


class IndicatorStrategy(BaseStrategy):
    strategy_category = "indicator"

    default_params = {
    }

    param_space = {
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def  generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        raise NotImplementedError

    def is_pattern(self, klines: List[dict], context: MarketContext) -> bool:
        raise NotImplementedError


# ========================
# 均值回归
# ========================
class BollingerBandStrategy(IndicatorStrategy):
    default_params = {
        "window": 20,
        "k": 2.0
    }

    param_space = {
        "window": [10, 20, 30],
        "k": [1.5, 2.0, 2.5]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.k = self.params["k"]
        self.prices = []

    def is_pattern(self, close: float, ma: float, upper: float, lower: float) -> Optional[SignalType]:
        """
        判断当前价格是否突破布林带，产生买入或卖出信号。
        """
        if close < lower:
            return SignalType.BUY
        elif close > upper:
            return SignalType.SELL
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        """
        根据最新的K线数据生成布林带买卖信号。
        """
        if not context or len(context.recent_klines) < self.window:
            return None
        
        # 计算最近的收盘价格
        close_price = context.recent_klines[-1]["close"]
        self.prices.append(close_price)
        
        if len(self.prices) < self.window:
            return None

        # 计算布林带
        ma = simple_moving_average(self.prices, self.window)
        std = compute_std(self.prices, self.window, ma)
        upper = ma + self.k * std
        lower = ma - self.k * std
        
       # 计算信号强度
        distance_from_upper = close_price - upper
        distance_from_lower = lower - close_price
        max_distance = max(distance_from_upper, distance_from_lower)
        strength = min(max_distance / (upper - lower), 2.0)  # 最大强度为2

        # 判断是否符合买卖信号
        signal_type = self.is_pattern(close_price, ma, upper, lower)

        if signal_type:
            return self.build_signal(
                context.recent_klines[-1], signal_type, {
                    "ma": ma,
                    "lower": lower,
                    "upper": upper,
                    "std": std
                }
            )
        return None


# ========================
# 动量类
# ========================
class RSIStrategy(IndicatorStrategy):
    default_params = {
        "period": 14,
        "overbought": 70,
        "oversold": 30
    }

    param_space = {
        "period": [10, 14, 20],
        "overbought": [60, 70, 80],
        "oversold": [20, 30, 40]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.period = self.params["period"]
        self.overbought = self.params["overbought"]
        self.oversold = self.params["oversold"]
        self.prices = []

    def is_pattern(self, rsi: float) -> Optional[SignalType]:
        if rsi < self.oversold:
            return SignalType.BUY
        elif rsi > self.overbought:
            return SignalType.SELL
        return None

    def calculate_signal_strength(self, rsi: float) -> float:
        """
        根据 RSI 计算信号强度，越接近超买/超卖的值，强度越高
        """
        if rsi < self.oversold:
            return min((self.oversold - rsi) / 30, 1.0)  # 最大强度为1
        elif rsi > self.overbought:
            return min((rsi - self.overbought) / 30, 1.0)  # 最大强度为1
        return 0.0

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        # 从 MarketContext 获取市场数据
        if not context or len(context.recent_klines) < self.period + 1:
            return None

        close_price = context.recent_klines[-1]["close"]
        self.prices.append(close_price)

        if len(self.prices) < self.period + 1:
            return None

        rsi = compute_rsi(self.prices, self.period)
        signal_type = self.is_pattern(rsi)

        if signal_type:
            strength = self.calculate_signal_strength(rsi)
            return self.build_signal(context.recent_klines[-1], signal_type, strength, {"rsi": rsi})
        return None


class ATRStrategy(IndicatorStrategy):
    default_params = {
        "window": 14,
        "atr_threshold": 2.0
    }

    param_space = {
        "window": [10, 14, 20],
        "atr_threshold": [1.5, 2.0, 2.5]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.atr_threshold = self.params["atr_threshold"]
        self.trs, self.prev_close = [], None

    def calculate_signal_strength(self, atr: float) -> float:
        """
        根据 ATR 与阈值的关系计算信号强度
        """
        return min(atr / self.atr_threshold, 1.0)  # 最大强度为1

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        # 从 MarketContext 获取市场数据
        if not context or len(context.recent_klines) < self.window:
            return None

        high, low, close = context.recent_klines[-1]["high"], context.recent_klines[-1]["low"], context.recent_klines[-1]["close"]
        if self.prev_close is None:
            self.prev_close = close
            return None

        tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
        self.trs.append(tr)
        self.prev_close = close

        if len(self.trs) < self.window:
            return None
        atr = sum(self.trs[-self.window:]) / self.window

        # 计算信号强度
        strength = self.calculate_signal_strength(atr)

        metadata = {"atr": atr}
        # 假设作为动量过滤器，不直接给出交易信号，仅用于辅助其他信号判断
        return self.build_signal(context.recent_klines[-1], SignalType.FILTER, strength, metadata)


# ========================
# 多空类
# ========================
class MACDStrategy(IndicatorStrategy):
    default_params = {
        "short_period": 12,
        "long_period": 26,
        "signal_period": 9
    }

    param_space = {
        "short_period": [8, 12, 16],
        "long_period": [20, 26, 30],
        "signal_period": [5, 9, 12]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.short_period = self.params["short_period"]
        self.long_period = self.params["long_period"]
        self.signal_period = self.params["signal_period"]
        self.prices = []

    def is_pattern(self, macd: float, signal: float, hist: float) -> Optional[SignalType]:
        if macd > signal and hist > 0:
            return SignalType.BUY
        elif macd < signal and hist < 0:
            return SignalType.SELL
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if not context or len(context.recent_klines) < max(self.short_period, self.long_period, self.signal_period):
            return None
        
        self.prices.append(context.recent_klines[-1]["close"])
        
        # 计算 MACD
        macd, signal, hist = compute_macd(self.prices, self.short_period, self.long_period, self.signal_period)
        
        # 信号强度定义
        strength = abs(hist) / max(abs(macd), abs(signal), 1.0)

        # 判断是否符合买卖信号
        signal_type = self.is_pattern(macd, signal, hist)

        if signal_type:
            return self.build_signal(
                context.recent_klines[-1], signal_type, strength, {
                    "macd": macd,
                    "signal": signal,
                    "hist": hist
                }
            )
        return None


# ========================
# 趋势类
# ========================
class SMACrossoverStrategy(IndicatorStrategy):
    default_params = {
        "short_window": 5,
        "long_window": 20
    }

    param_space = {
        "short_window": [5, 10, 15],
        "long_window": [20, 30, 40]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.short_window = self.params["short_window"]
        self.long_window = self.params["long_window"]
        self.prices: List[float] = []
        self.last_signal: Optional[SignalType] = None

    def is_pattern(self, short_ma: float, long_ma: float) -> Optional[SignalType]:
        if short_ma > long_ma and self.last_signal != SignalType.BUY:
            return SignalType.BUY
        elif short_ma < long_ma and self.last_signal != SignalType.SELL:
            return SignalType.SELL
        return None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if not context or len(context.recent_klines) < self.long_window:
            return None
        
        self.prices.append(context.recent_klines[-1]["close"])

        # 计算移动均线
        short_ma = simple_moving_average(self.prices, self.short_window)
        long_ma = simple_moving_average(self.prices, self.long_window)
        signal_type = self.is_pattern(short_ma, long_ma)

        if signal_type:
            self.last_signal = signal_type
            return self.build_signal(
                context.recent_klines[-1], signal_type, {"short_ma": short_ma, "long_ma": long_ma}
            )
        return None


class EMACrossoverStrategy(IndicatorStrategy):
    default_params = {
        "short_window": 5,
        "long_window": 20
    }

    param_space = {
        "short_window": [5, 10, 15],
        "long_window": [20, 30, 40]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.short_window = self.params["short_window"]
        self.long_window = self.params["long_window"]
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

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if not context or len(context.recent_klines) < self.long_window:
            return None
        
        self.prices.append(context.recent_klines[-1]["close"])

        short_ema = exponential_moving_average(self.prices, self.short_window, self.prev_short_ema)
        long_ema = exponential_moving_average(self.prices, self.long_window, self.prev_long_ema)

        self.prev_short_ema = short_ema
        self.prev_long_ema = long_ema

        signal_type = self.is_pattern(short_ema, long_ema)

        if signal_type:
            self.last_signal = signal_type
            return self.build_signal(
                context.recent_klines[-1], signal_type, {"short_ema": short_ema, "long_ema": long_ema}
            )
        return None


class ADXStrategy(IndicatorStrategy):
    default_params = {
        "window": 14,
        "adx_threshold": 25.0
    }

    param_space = {
        "window": [10, 14, 20],
        "adx_threshold": [20.0, 25.0, 30.0]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.adx_threshold = self.params["adx_threshold"]
        self.highs, self.lows, self.closes = [], [], []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if not context or  len(context.recent_klines) < 1:
            return None

        self.highs.append(context.recent_klines[-1]["high"])
        self.lows.append(context.recent_klines[-1]["low"])
        self.closes.append(context.recent_klines[-1]["close"])

        if len(self.highs) <= self.window:
            return None

        # 计算 ADX（此处仅用模拟值）
        adx = self.mock_adx_calc()
        metadata = {"adx": adx}
        if adx > self.adx_threshold:
            return self.build_signal(
                context.recent_klines[-1], SignalType.STRONG_TREND, {"adx": adx}
            )
        return None

    def mock_adx_calc(self) -> float:
        return 20 + 10 * ((len(self.closes) % 10) / 10)  # 模拟值用于占位


class CCIStrategy(IndicatorStrategy):
    default_params = {
        "window": 20,
        "threshold": 100.0
    }

    param_space = {
        "window": [10, 20, 30],
        "threshold": [50.0, 100.0, 150.0]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.threshold = self.params["threshold"]
        self.typical_prices = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if not context or  len(context.recent_klines) < 1:
            return None

        tp = (context.recent_klines[-1]["high"] + context.recent_klines[-1]["low"] + context.recent_klines[-1]["close"]) / 3
        self.typical_prices.append(tp)

        if len(self.typical_prices) < self.window:
            return None

        ma = simple_moving_average(self.typical_prices, self.window)
        md = compute_std(self.typical_prices, self.window, ma)
        if md == 0:
            return None
        cci = (tp - ma) / (0.015 * md)

        metadata = {"cci": cci, "ma": ma, "md": md}
        if cci < -self.threshold:
            return self.build_signal(context.recent_klines[-1], SignalType.BUY, metadata)
        elif cci > self.threshold:
            return self.build_signal(context.recent_klines[-1], SignalType.SELL, metadata)
        return None


class StochasticStrategy(IndicatorStrategy):
    default_params = {
        "k_period": 14,
        "d_period": 3
    }

    param_space = {
        "k_period": [10, 14, 20],
        "d_period": [2, 3, 5]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.k_period = self.params["k_period"]
        self.d_period = self.params["d_period"]
        self.highs, self.lows, self.ks = [], [], []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if not context or  len(context.recent_klines) < 1:
            return None

        # 从 MarketContext 获取数据
        kline = context.recent_klines[-1]
        self.highs.append(kline["high"])
        self.lows.append(kline["low"])

        if len(self.highs) < self.k_period:
            return None

        recent_high = max(self.highs[-self.k_period:])
        recent_low = min(self.lows[-self.k_period:])
        if recent_high == recent_low:
            return None

        # 计算 RSV 和 %K
        rsv = (kline["close"] - recent_low) / (recent_high - recent_low) * 100
        k = 2/3 * self.ks[-1] + 1/3 * rsv if self.ks else rsv
        self.ks.append(k)

        # 计算 %D
        d = sum(self.ks[-self.d_period:]) / min(len(self.ks), self.d_period)

        # 信号强度：例如使用 %K 和 %D 的差作为强度
        signal_strength = abs(k - d)

        # 构建信号
        metadata = {"k": k, "d": d, "signal_strength": signal_strength}
        
        if k < 20 and d < 20 and k > d:
            return self.build_signal(kline, SignalType.BUY, metadata, strength=signal_strength)
        elif k > 80 and d > 80 and k < d:
            return self.build_signal(kline, SignalType.SELL, metadata, strength=signal_strength)
        
        return None
