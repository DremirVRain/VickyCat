from typing import Optional
from datetime import datetime
from indicator_strategy import IndicatorStrategy
from base_strategy import MarketContext
from strategy_signal import Signal, SignalType


class VolumeBreakoutStrategy(IndicatorStrategy):
    """
    通过突破平均成交量与某个阈值，判断成交量放大信号。
    """
    default_params = {
        "window": 20,
        "threshold": 1.5
    }

    param_space = {
        "window": [10, 20, 30],
        "threshold": [1.0, 1.5, 2.0]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.threshold = self.params["threshold"]
        self.volumes = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
        volume = kline.get("volume") or kline.get("amount", 0)
        self.volumes.append(volume)
        if len(self.volumes) < self.window:
            return None

        avg_volume = sum(self.volumes[-self.window:]) / self.window
        metadata = {"volume": volume, "avg_volume": avg_volume}
        if volume > avg_volume * self.threshold:
            return self.build_signal(kline, SignalType.BUY, metadata)
        return None


class OBVStrategy(IndicatorStrategy):
    """
    通过OBV（On-Balance Volume）指标生成信号。
    """
    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = kwargs
        self.obv = 0
        self.prev_close = None

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
        close, volume = kline["close"], kline.get("volume") or kline.get("amount", 0)
        if self.prev_close is None:
            self.prev_close = close
            return None

        if close > self.prev_close:
            self.obv += volume
        elif close < self.prev_close:
            self.obv -= volume

        self.prev_close = close
        metadata = {"obv": self.obv}
        return self.build_signal(kline, SignalType.INDICATOR, metadata)


class VolumePullbackStrategy(IndicatorStrategy):
    """
    缩量回调策略：当前成交量低于过去window均值的某个比例threshold_pullback，且价格较之前的高点出现回调时，给出缩量回调信号。
    """

    default_params = {
        "window": 20,
        "threshold_pullback": 0.7,
        "price_pullback_ratio": 0.02
    }

    param_space = {
        "window": [10, 20, 30],
        "threshold_pullback": [0.5, 0.7, 0.9],
        "price_pullback_ratio": [0.01, 0.02, 0.05]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.threshold_pullback = self.params["threshold_pullback"]
        self.price_pullback_ratio = self.params["price_pullback_ratio"]
        self.volumes = []
        self.highs = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        kline = context.recent_klines[-1]
        volume = kline.get("volume") or kline.get("amount", 0)
        high = kline["high"]
        close = kline["close"]

        self.volumes.append(volume)
        self.highs.append(high)

        if len(self.volumes) < self.window:
            return None

        avg_volume = sum(self.volumes[-self.window:]) / self.window
        recent_high = max(self.highs[-self.window:])

        volume_condition = volume < avg_volume * self.threshold_pullback
        price_condition = close < recent_high * (1 - self.price_pullback_ratio)

        if volume_condition and price_condition:
            metadata = {
                "volume": volume,
                "avg_volume": avg_volume,
                "recent_high": recent_high,
                "price_pullback_ratio": self.price_pullback_ratio,
            }
            return self.build_signal(kline, SignalType.SELL, metadata)
        return None

