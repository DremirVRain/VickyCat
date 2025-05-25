from typing import Optional
from datetime import datetime
from indicator_strategy import IndicatorStrategy
from strategy_signal import Signal, SignalType


class VolumeBreakoutStrategy(IndicatorStrategy):
    def __init__(self, symbol: str, window: int = 20, threshold: float = 1.5):
        super().__init__(symbol)
        self.window = window
        self.threshold = threshold
        self.volumes = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        volume = kline.get("volume") or kline.get("amount", 0)
        self.volumes.append(volume)
        if len(self.volumes) < self.window:
            return None

        avg_volume = sum(self.volumes[-self.window:]) / self.window
        metadata = {"volume": volume, "avg_volume": avg_volume}
        if volume > avg_volume * self.threshold:
            return self.build_signal(kline, SignalType.BUY, metadata)
        return None

    def get_params(self) -> dict:
        return {
            "window": self.window,
            "threshold": self.threshold,
        }

    def set_params(self, **params):
        self.window = params.get("window", self.window)
        self.threshold = params.get("threshold", self.threshold)


class OBVStrategy(IndicatorStrategy):
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.obv = 0
        self.prev_close = None

    def generate_signal(self, kline: dict) -> Optional[Signal]:
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
        return Signal(
            symbol=self.symbol,
            signal_type=SignalType.INDICATOR,
            time=kline["timestamp"],
            metadata=metadata,
        )

    def get_params(self) -> dict:
        return {}

    def set_params(self, **params):
        pass


class VolumePullbackStrategy(IndicatorStrategy):
    """
    缩量回调策略：
    当当前成交量低于过去window均值的某个比例threshold_pullback，
    且价格较之前的高点出现回调时，给出缩量回调信号。
    """

    def __init__(
        self,
        symbol: str,
        window: int = 20,
        threshold_pullback: float = 0.7,
        price_pullback_ratio: float = 0.02,
    ):
        super().__init__(symbol)
        self.window = window
        self.threshold_pullback = threshold_pullback  # 成交量缩量阈值，0.7 表示当前量<70%均量
        self.price_pullback_ratio = price_pullback_ratio  # 价格回调比例，比如2%
        self.volumes = []
        self.highs = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
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

    def get_params(self) -> dict:
        return {
            "window": self.window,
            "threshold_pullback": self.threshold_pullback,
            "price_pullback_ratio": self.price_pullback_ratio,
        }

    def set_params(self, **params):
        self.window = params.get("window", self.window)
        self.threshold_pullback = params.get("threshold_pullback", self.threshold_pullback)
        self.price_pullback_ratio = params.get("price_pullback_ratio", self.price_pullback_ratio)
