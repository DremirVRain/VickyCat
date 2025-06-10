from base_strategy import BaseStrategy, MarketContext
from strategy_signal import SignalType, Signal
from typing import Optional
from datetime import datetime


class BreakoutStrategy(BaseStrategy):
    """
    判断是否突破最近 N 根 K 线的高点或低点，用于趋势确认。
    """
    default_params = {
        "window": 20,
        "max_length": 1000
    }

    param_space = {
        "window": [10, 20, 30],
        "max_length": [500, 1000, 1500]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.max_length = self.params["max_length"]
        self.highs = []
        self.lows = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]  # 获取最新K线数据
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


class MarketStructureStrategy(BaseStrategy):
    """
    识别前高前低（HH/HL, LH/LL）结构，辅助趋势识别与转折判断。
    """
    default_params = {
        "max_length": 200
    }

    param_space = {
        "max_length": [100, 200, 300]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.max_length = self.params["max_length"]
        self.highs = []
        self.lows = []
        self.prev_high = None
        self.prev_low = None
        self.prev_trend = None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
        high = kline["high"]
        low = kline["low"]
        close = kline["close"]
        
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


class HeadShouldersStrategy(BaseStrategy):
    """
    识别典型的头肩顶（Head & Shoulders）或头肩底形态，提示趋势反转。
    """
    default_params = {
        "window": 50,
        "tolerance": 0.03
    }

    param_space = {
        "window": [30, 50, 100],
        "tolerance": [0.02, 0.03, 0.05]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.tolerance = self.params["tolerance"]
        self.klines = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
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
        right = highs[mid + 1:]
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
        right_low = min(lows[mid + 1:])

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


class TripleTopBottomStrategy(BaseStrategy):
    """
    识别三顶/三底结构：形成三个相近的高点或低点，第三个未突破前两个。
    """
    default_params = {
        "tolerance": 0.005,
        "window": 7
    }

    param_space = {
        "tolerance": [0.003, 0.005, 0.01],
        "window": [5, 7, 10]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.tolerance = self.params["tolerance"]
        self.window = self.params["window"]
        self.recent_highs = []
        self.recent_lows = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
        self.recent_highs.append(kline['high'])
        self.recent_lows.append(kline['low'])

        if len(self.recent_highs) < self.window:
            return None

        last_highs = self.recent_highs[-self.window:]
        last_lows = self.recent_lows[-self.window:]

        h1, h2, h3 = last_highs[-5], last_highs[-3], last_highs[-1]
        if abs(h1 - h2) / h1 < self.tolerance and abs(h2 - h3) / h2 < self.tolerance and h3 < last_highs[-4]:
            metadata = {"pattern": "TripleTop", "high1": h1, "high2": h2, "high3": h3}
            return self.build_signal(kline, SignalType.SELL, metadata)

        l1, l2, l3 = last_lows[-5], last_lows[-3], last_lows[-1]
        if abs(l1 - l2) / l1 < self.tolerance and abs(l2 - l3) / l2 < self.tolerance and l3 > last_lows[-4]:
            metadata = {"pattern": "TripleBottom", "low1": l1, "low2": l2, "low3": l3}
            return self.build_signal(kline, SignalType.BUY, metadata)

        return None


class DoubleTopBottomStrategy(BaseStrategy):
    """
    识别双顶/双底结构：连续出现两个相近高点或低点，第二个未能突破。
    """
    default_params = {
        "tolerance": 0.005,
        "window": 5
    }

    param_space = {
        "tolerance": [0.003, 0.005, 0.01],
        "window": [3, 5, 7]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.tolerance = self.params["tolerance"]
        self.window = self.params["window"]
        self.recent_highs = []
        self.recent_lows = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
        high, low = kline['high'], kline['low']
        self.recent_highs.append(high)
        self.recent_lows.append(low)

        if len(self.recent_highs) < self.window:
            return None

        last_highs = self.recent_highs[-self.window:]
        last_lows = self.recent_lows[-self.window:]

        h1, h2 = last_highs[-3], last_highs[-1]
        if abs(h1 - h2) / h1 < self.tolerance and h2 < last_highs[-2]:
            return self.build_signal(kline, SignalType.SELL, {"pattern": "DoubleTop"})

        l1, l2 = last_lows[-3], last_lows[-1]
        if abs(l1 - l2) / l1 < self.tolerance and l2 > last_lows[-2]:
            return self.build_signal(kline, SignalType.BUY, {"pattern": "DoubleBottom"})

        return None


class TripleTopBottomStrategy(BaseStrategy):
    """
    识别三顶/三底结构：形成三个相近的高点或低点，第三个未突破前两个。
    """
    default_params = {
        "tolerance": 0.005,
        "window": 7
    }

    param_space = {
        "tolerance": [0.003, 0.005, 0.01],
        "window": [5, 7, 10]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.tolerance = self.params["tolerance"]
        self.window = self.params["window"]
        self.recent_highs = []
        self.recent_lows = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
        self.recent_highs.append(kline['high'])
        self.recent_lows.append(kline['low'])

        if len(self.recent_highs) < self.window:
            return None

        last_highs = self.recent_highs[-self.window:]
        last_lows = self.recent_lows[-self.window:]

        h1, h2, h3 = last_highs[-5], last_highs[-3], last_highs[-1]
        if abs(h1 - h2) / h1 < self.tolerance and abs(h2 - h3) / h2 < self.tolerance and h3 < last_highs[-4]:
            metadata = {"pattern": "TripleTop", "high1": h1, "high2": h2, "high3": h3}
            return self.build_signal(kline, SignalType.SELL, metadata)

        l1, l2, l3 = last_lows[-5], last_lows[-3], last_lows[-1]
        if abs(l1 - l2) / l1 < self.tolerance and abs(l2 - l3) / l2 < self.tolerance and l3 > last_lows[-4]:
            metadata = {"pattern": "TripleBottom", "low1": l1, "low2": l2, "low3": l3}
            return self.build_signal(kline, SignalType.BUY, metadata)

        return None


class FlagPennantStrategy(BaseStrategy):
    """
    识别旗形/三角形形态：价格形成一个整理的三角形区域，突破该区域可能会继续沿着趋势方向运动。
    """
    default_params = {
        "window": 20,
        "threshold": 0.03
    }

    param_space = {
        "window": [15, 20, 30],
        "threshold": [0.01, 0.03, 0.05]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.threshold = self.params["threshold"]
        self.highs = []
        self.lows = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
        self.highs.append(kline['high'])
        self.lows.append(kline['low'])

        if len(self.highs) < self.window:
            return None

        recent_highs = self.highs[-self.window:]
        recent_lows = self.lows[-self.window:]

        flag_high = max(recent_highs)
        flag_low = min(recent_lows)

        if (flag_high - flag_low) / flag_low < self.threshold:
            metadata = {
                "pattern": "FlagPennant",
                "flag_high": flag_high,
                "flag_low": flag_low
            }
            return self.build_signal(kline, SignalType.INDICATOR, metadata)

        return None


class GapStrategy(BaseStrategy):
    """
    识别跳空：当前开盘价与前一根K线的收盘价差距较大。
    """
    default_params = {
        "threshold": 0.02
    }

    param_space = {
        "threshold": [0.01, 0.02, 0.05]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.threshold = self.params["threshold"]
        self.prev_close = None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
        close = kline['close']
        open_price = kline['open']

        if self.prev_close is None:
            self.prev_close = close
            return None

        gap_up = open_price > self.prev_close * (1 + self.threshold)
        gap_down = open_price < self.prev_close * (1 - self.threshold)

        if gap_up:
            metadata = {"pattern": "GapUp", "gap": open_price - self.prev_close}
            return self.build_signal(kline, SignalType.BUY, metadata)
        elif gap_down:
            metadata = {"pattern": "GapDown", "gap": self.prev_close - open_price}
            return self.build_signal(kline, SignalType.SELL, metadata)

        self.prev_close = close
        return None


class RangeBoundStrategy(BaseStrategy):
    """
    判断价格是否处于区间震荡：高点低点相对稳定。
    """
    default_params = {
        "window": 20,
        "threshold": 0.01
    }

    param_space = {
        "window": [10, 20, 30],
        "threshold": [0.005, 0.01, 0.02]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.threshold = self.params["threshold"]
        self.highs = []
        self.lows = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
        self.highs.append(kline['high'])
        self.lows.append(kline['low'])

        if len(self.highs) < self.window:
            return None

        recent_highs = self.highs[-self.window:]
        recent_lows = self.lows[-self.window:]

        range_high = max(recent_highs)
        range_low = min(recent_lows)

        if (range_high - range_low) / range_low < self.threshold:
            metadata = {
                "pattern": "RangeBound",
                "range_high": range_high,
                "range_low": range_low
            }
            return self.build_signal(kline, SignalType.INDICATOR, metadata)
        return None


class SupportResistanceBreakStrategy(BaseStrategy):
    """
    判断价格突破支撑/阻力。
    """
    default_params = {
        "window": 20
    }

    param_space = {
        "window": [10, 20, 30]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.closes = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
        close = kline['close']
        self.closes.append(close)
        
        if len(self.closes) < self.window:
            return None

        recent_closes = self.closes[-self.window:]
        support = min(recent_closes)
        resistance = max(recent_closes)

        if close > resistance:
            metadata = {
                "pattern": "BreakResistance",
                "resistance": resistance
            }
            return self.build_signal(kline, SignalType.BUY, metadata)
        elif close < support:
            metadata = {
                "pattern": "BreakSupport",
                "support": support
            }
            return self.build_signal(kline, SignalType.SELL, metadata)
        return None


class CupWithHandleStrategy(BaseStrategy):
    """
    识别“杯柄形态”，以趋势持续的回撤-盘整-上破逻辑为核心。
    """
    default_params = {
        "cup_length": 15,
        "handle_length": 5
    }

    param_space = {
        "cup_length": [10, 15, 20],
        "handle_length": [3, 5, 7]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.cup_length = self.params["cup_length"]
        self.handle_length = self.params["handle_length"]
        self.closes = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
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
    default_params = {
        "window": 20
    }

    param_space = {
        "window": [10, 20, 30]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.closes = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
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
    default_params = {
        "window": 20
    }

    param_space = {
        "window": [10, 20, 30]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.highs = []
        self.lows = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
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
