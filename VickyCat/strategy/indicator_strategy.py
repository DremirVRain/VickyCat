# strategies/indicator.py
from base_strategy import BaseStrategy
from signal import Signal, SignalType
from typing import Optional
from datetime import datetime


class IndicatorStrategy(BaseStrategy):
    def generate_signal(self, kline: dict) -> Optional[Signal]:
        raise NotImplementedError


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

        recent = self.prices[-self.window:]
        ma = sum(recent) / self.window
        std = (sum((p - ma) ** 2 for p in recent) / self.window) ** 0.5

        upper = ma + self.k * std
        lower = ma - self.k * std

        if kline["close"] < lower:
            return Signal(
                symbol=self.symbol,
                timestamp=datetime.strptime(kline["timestamp"], "%Y-%m-%d %H:%M:%S"),
                signal_type=SignalType.BUY,
                strength=1.0,
                strategy_name=self.__class__.__name__,
                metadata={"ma": ma, "lower": lower, "upper": upper},
            )

        elif kline["close"] > upper:
            return Signal(
                symbol=self.symbol,
                timestamp=datetime.strptime(kline["timestamp"], "%Y-%m-%d %H:%M:%S"),
                signal_type=SignalType.SELL,
                strength=1.0,
                strategy_name=self.__class__.__name__,
                metadata={"ma": ma, "lower": lower, "upper": upper},
            )

        return None
