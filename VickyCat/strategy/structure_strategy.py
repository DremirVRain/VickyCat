from base_strategy import BaseStrategy
from strategy_signal import SignalType, Signal
from typing import Optional
from datetime import datetime


class BreakoutStrategy(BaseStrategy):
    """
    判断是否突破最近 N 根 K 线的高点或低点，用于趋势确认。
    """
    def __init__(self, symbol: str, window: int = 20, max_length: int = 1000):
        super().__init__(symbol)
        self.window = window
        self.max_length = max_length
        self.highs = []
        self.lows = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.highs.append(kline["high"])
        self.lows.append(kline["low"])

        # 限制历史长度，防止内存无限增长
        if len(self.highs) > self.max_length:
            self.highs = self.highs[-self.max_length:]
            self.lows = self.lows[-self.max_length:]

        if len(self.highs) < self.window:
            return None

        recent_high = max(self.highs[-self.window:])
        recent_low = min(self.lows[-self.window:])

        metadata = {"recent_high": recent_high, "recent_low": recent_low}

        if kline["close"] > recent_high:
            return self.build_signal(kline, SignalType.BUY, metadata)

        elif kline["close"] < recent_low:
            return self.build_signal(kline, SignalType.SELL, metadata)

        return None

    def get_params(self) -> dict:
        return {
            "window": self.window,
            "max_length": self.max_length,
        }

    def set_params(self, **params):
        self.window = params.get("window", self.window)
        self.max_length = params.get("max_length", self.max_length)


class MarketStructureStrategy(BaseStrategy):
    """
    识别前高前低（HH/HL, LH/LL）结构，辅助趋势识别与转折判断。
    """
    def __init__(self, symbol: str, max_length: int = 200):
        super().__init__(symbol)
        self.max_length = max_length
        self.highs = []
        self.lows = []
        self.prev_high = None
        self.prev_low = None
        self.prev_trend = None

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        high = kline["high"]
        low = kline["low"]
        close = kline["close"]
        timestamp = kline["timestamp"]

        self.highs.append(high)
        self.lows.append(low)

        if len(self.highs) > self.max_length:
            self.highs = self.highs[-self.max_length:]
            self.lows = self.lows[-self.max_length:]

        if self.prev_high is None or self.prev_low is None:
            self.prev_high = high
            self.prev_low = low
            return None

        structure = "range"
        if high > self.prev_high and low > self.prev_low:
            structure = "uptrend"
        elif high < self.prev_high and low < self.prev_low:
            structure = "downtrend"

        metadata = {
            "structure": structure,
            "prev_high": self.prev_high,
            "prev_low": self.prev_low,
            "current_high": high,
            "current_low": low,
        }

        self.prev_high = high
        self.prev_low = low
        self.prev_trend = structure

        return self.build_signal(kline, SignalType.STRUCTURE, metadata)

    def get_params(self) -> dict:
        return {
            "max_length": self.max_length,
        }

    def set_params(self, **params):
        self.max_length = params.get("max_length", self.max_length)

class HeadShouldersStrategy(BaseStrategy):
    """
    识别典型的头肩顶（Head & Shoulders）或头肩底形态，提示趋势反转。
    """
    def __init__(self, symbol: str, window: int = 50, tolerance: float = 0.03):
        super().__init__(symbol)
        self.window = window
        self.tolerance = tolerance
        self.klines = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.klines.append(kline)
        if len(self.klines) < self.window:
            return None

        # 只保留 window 个k线
        self.klines = self.klines[-self.window:]
        highs = [k["high"] for k in self.klines]
        lows = [k["low"] for k in self.klines]

        # 简化版头肩顶检测逻辑
        mid = len(highs) // 2
        left = highs[:mid]
        right = highs[mid+1:]
        head = highs[mid]

        if not left or not right:
            return None

        left_shoulder = max(left)
        right_shoulder = max(right)

        # 判断是否形成头肩顶结构
        if (
            head > left_shoulder * (1 + self.tolerance) and
            head > right_shoulder * (1 + self.tolerance) and
            abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder) < self.tolerance
        ):
            # 触发头肩顶信号
            metadata = {
                "head": head,
                "left_shoulder": left_shoulder,
                "right_shoulder": right_shoulder
            }
            return self.build_signal(kline, SignalType.SELL, metadata)

        # 反过来是头肩底
        mid_low = lows[mid]
        left_low = min(lows[:mid])
        right_low = min(lows[mid+1:])

        if (
            mid_low < left_low * (1 - self.tolerance) and
            mid_low < right_low * (1 - self.tolerance) and
            abs(left_low - right_low) / max(left_low, right_low) < self.tolerance
        ):
            metadata = {
                "head": mid_low,
                "left_shoulder": left_low,
                "right_shoulder": right_low
            }
            return self.build_signal(kline, SignalType.BUY, metadata)

        return None

    def get_params(self) -> dict:
        return {
            "window": self.window,
            "tolerance": self.tolerance
        }

    def set_params(self, **params):
        self.window = params.get("window", self.window)
        self.tolerance = params.get("tolerance", self.tolerance)

class DoubleTopBottomStrategy(BaseStrategy):
    """
    识别双顶/双底结构：连续出现两个相近高点或低点，第二个未能突破。
    """
    def __init__(self, symbol: str, tolerance: float = 0.005):
        super().__init__(symbol)
        self.recent_highs = []
        self.recent_lows = []
        self.tolerance = tolerance

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        high, low = kline['high'], kline['low']
        self.recent_highs.append(high)
        self.recent_lows.append(low)

        if len(self.recent_highs) < 5:
            return None

        last_highs = self.recent_highs[-5:]
        last_lows = self.recent_lows[-5:]

        h1, h2 = last_highs[-3], last_highs[-1]
        if abs(h1 - h2) / h1 < self.tolerance and h2 < last_highs[-2]:
            return self.build_signal(kline, SignalType.SELL, {"pattern": "DoubleTop"})

        l1, l2 = last_lows[-3], last_lows[-1]
        if abs(l1 - l2) / l1 < self.tolerance and l2 > last_lows[-2]:
            return self.build_signal(kline, SignalType.BUY, {"pattern": "DoubleBottom"})

        return None


class RangeBoundStrategy(BaseStrategy):
    """
    判断价格是否处于区间震荡：高点低点相对稳定。
    """
    def __init__(self, symbol: str, window: int = 20, threshold: float = 0.01):
        super().__init__(symbol)
        self.window = window
        self.threshold = threshold
        self.highs = []
        self.lows = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.highs.append(kline['high'])
        self.lows.append(kline['low'])

        if len(self.highs) < self.window:
            return None

        recent_highs = self.highs[-self.window:]
        recent_lows = self.lows[-self.window:]

        range_high = max(recent_highs)
        range_low = min(recent_lows)

        if (range_high - range_low) / range_low < self.threshold:
            return self.build_signal(kline, SignalType.INDICATOR, {
                "pattern": "RangeBound",
                "range_high": range_high,
                "range_low": range_low
            })
        return None

class SupportResistanceBreakStrategy(BaseStrategy):
    """
    判断价格突破支撑/阻力。
    """
    def __init__(self, symbol: str, window: int = 20):
        super().__init__(symbol)
        self.window = window
        self.closes = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        close = kline['close']
        self.closes.append(close)
        if len(self.closes) < self.window:
            return None

        recent_closes = self.closes[-self.window:]
        support = min(recent_closes)
        resistance = max(recent_closes)

        if close > resistance:
            return self.build_signal(kline, SignalType.BUY, {
                "pattern": "BreakResistance",
                "resistance": resistance
            })
        elif close < support:
            return self.build_signal(kline, SignalType.SELL, {
                "pattern": "BreakSupport",
                "support": support
            })
        return None


class FlagPennantStrategy(BaseStrategy):
    """
    识别旗形/三角旗形整理后的突破
    """
    def __init__(self, symbol: str, trend_window: int = 10, consolidation_window: int = 5):
        super().__init__(symbol)
        self.trend_window = trend_window
        self.consolidation_window = consolidation_window
        self.closes = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.closes.append(kline["close"])
        if len(self.closes) < self.trend_window + self.consolidation_window:
            return None

        trend_part = self.closes[-(self.trend_window + self.consolidation_window):-self.consolidation_window]
        consolidation_part = self.closes[-self.consolidation_window:]

        trend_direction = trend_part[-1] - trend_part[0]
        consolidation_range = max(consolidation_part) - min(consolidation_part)

        metadata = {
            "trend_direction": trend_direction,
            "consolidation_range": consolidation_range
        }

        if abs(trend_direction) > consolidation_range * 2:
            if trend_direction > 0 and kline["close"] > max(consolidation_part):
                return self.build_signal(kline, SignalType.BUY, metadata)
            elif trend_direction < 0 and kline["close"] < min(consolidation_part):
                return self.build_signal(kline, SignalType.SELL, metadata)

        return None


class CupWithHandleStrategy(BaseStrategy):
    """
    识别“杯柄形态”，以趋势持续的回撤-盘整-上破逻辑为核心。
    """
    def __init__(self, symbol: str, cup_length: int = 15, handle_length: int = 5):
        super().__init__(symbol)
        self.cup_length = cup_length
        self.handle_length = handle_length
        self.closes = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.closes.append(kline["close"])
        total_len = self.cup_length + self.handle_length
        if len(self.closes) < total_len:
            return None

        cup = self.closes[-total_len:-self.handle_length]
        handle = self.closes[-self.handle_length:]

        cup_min = min(cup)
        cup_max = max(cup)
        handle_max = max(handle)
        handle_min = min(handle)

        metadata = {
            "cup_min": cup_min,
            "cup_max": cup_max,
            "handle_max": handle_max
        }

        # 简化判断：识别较深回撤后企稳震荡上破
        if (cup_max - cup_min) > 0.05 * cup_max and (handle_max - handle_min) < 0.03 * cup_max:
            if kline["close"] > handle_max:
                return self.build_signal(kline, SignalType.BUY, metadata)

        return None


class TrendlineBreakStrategy(BaseStrategy):
    """
    识别趋势线突破，粗略估算线性趋势并判断突破信号
    """
    def __init__(self, symbol: str, window: int = 20):
        super().__init__(symbol)
        self.window = window
        self.closes = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.closes.append(kline["close"])
        if len(self.closes) < self.window:
            return None

        lows = self.closes[-self.window:]
        trendline = lows[0] + (lows[-1] - lows[0]) * (len(lows) - 1) / (self.window - 1)
        close = kline["close"]
        metadata = {"trendline_estimate": trendline, "close": close}

        if close > trendline * 1.01:
            return self.build_signal(kline, SignalType.BUY, metadata)
        elif close < trendline * 0.99:
            return self.build_signal(kline, SignalType.SELL, metadata)

        return None

class WedgeStrategy(BaseStrategy):
    """
    通过高低点收敛判断是否可能突破，区分上涨楔形与下跌楔形。
    """
    def __init__(self, symbol: str, window: int = 20):
        super().__init__(symbol)
        self.window = window
        self.highs = []
        self.lows = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.highs.append(kline["high"])
        self.lows.append(kline["low"])

        if len(self.highs) < self.window:
            return None

        max_high = max(self.highs[-self.window:])
        min_high = min(self.highs[-self.window:])
        max_low = max(self.lows[-self.window:])
        min_low = min(self.lows[-self.window:])

        narrowing = (max_high - min_high) + (max_low - min_low)
        if narrowing < (0.02 * (max_high + min_low)):
            if kline["close"] > max_high:
                return self.build_signal(kline, SignalType.BUY, {"type": "falling_wedge"})
            elif kline["close"] < min_low:
                return self.build_signal(kline, SignalType.SELL, {"type": "rising_wedge"})

        return None