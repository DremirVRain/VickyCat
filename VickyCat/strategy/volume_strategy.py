# strategies/volume.py
from base_strategy import BaseStrategy
from signal import Signal, SignalType
from typing import Optional
from datetime import datetime


class VolumeSpikeStrategy(BaseStrategy):
    def __init__(self, symbol: str, window: int = 20, multiplier: float = 2.0):
        super().__init__(symbol)
        self.window = window
        self.multiplier = multiplier
        self.volumes = []

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        self.volumes.append(kline["volume"])
        if len(self.volumes) < self.window:
            return None

        avg_volume = sum(self.volumes[-self.window:]) / self.window
        if kline["volume"] > self.multiplier * avg_volume:
            return Signal(
                symbol=self.symbol,
                timestamp=datetime.strptime(kline["timestamp"], "%Y-%m-%d %H:%M:%S"),
                signal_type=SignalType.BUY,
                strength=1.0,
                strategy_name=self.__class__.__name__,
                metadata={"volume": kline["volume"], "avg_volume": avg_volume},
            )
        return None

