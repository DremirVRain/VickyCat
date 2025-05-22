# strategies/structure.py
from base_strategy import BaseStrategy
from strategy_signal import Signal, SignalType
from typing import Optional
from datetime import datetime


class BreakoutStrategy(BaseStrategy):
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

        recent_high = max(self.highs[-self.window:])
        recent_low = min(self.lows[-self.window:])

        if kline["close"] > recent_high:
            return Signal(
                symbol=self.symbol,
                timestamp=datetime.strptime(kline["timestamp"], "%Y-%m-%d %H:%M:%S"),
                signal_type=SignalType.BUY,
                strength=1.0,
                strategy_name=self.__class__.__name__,
            )

        elif kline["close"] < recent_low:
            return Signal(
                symbol=self.symbol,
                timestamp=datetime.strptime(kline["timestamp"], "%Y-%m-%d %H:%M:%S"),
                signal_type=SignalType.SELL,
                strength=1.0,
                strategy_name=self.__class__.__name__,
            )

        return None
