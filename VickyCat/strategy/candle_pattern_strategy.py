# strategies/candle_pattern.py
from base_strategy import BaseStrategy
from signal import Signal, SignalType
from typing import Optional
from datetime import datetime

class CandlePatternStrategy(BaseStrategy):
    def is_pattern(self, *klines: dict) -> bool:
        raise NotImplementedError

class HammerPattern(CandlePatternStrategy):
    def is_pattern(self, kline: dict) -> bool:
        o, h, l, c = kline["open"], kline["high"], kline["low"], kline["close"]
        body = abs(c - o)
        lower_shadow = o - l if c >= o else c - l
        upper_shadow = h - c if c >= o else h - o
        return lower_shadow > 2 * body and upper_shadow < body

class HangingManPattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def is_pattern(self, kline: dict) -> bool:
        o, h, l, c = kline["open"], kline["high"], kline["low"], kline["close"]
        body = abs(c - o)
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)
        return lower_shadow > 2 * body and upper_shadow < body and c < o

class BullishEngulfingPattern(CandlePatternStrategy):
    signal_type = SignalType.BUY

    def is_pattern(self, kline: dict) -> bool:
        prev = kline.get("prev")
        if not prev:
            return False
        return (
            prev["open"] > prev["close"] and
            kline["open"] < kline["close"] and
            kline["open"] < prev["close"] and
            kline["close"] > prev["open"]
        )

class BearishEngulfingPattern(CandlePatternStrategy):
    signal_type = SignalType.SELL

    def is_pattern(self, kline: dict) -> bool:
        prev = kline.get("prev")
        if not prev:
            return False
        return (
            prev["open"] < prev["close"] and
            kline["open"] > kline["close"] and
            kline["open"] > prev["close"] and
            kline["close"] < prev["open"]
        )
    
class EngulfingPattern(CandlePatternStrategy):
    def is_pattern(self, curr: dict, prev: dict) -> bool:
        return (
            (prev["close"] < prev["open"] and curr["close"] > curr["open"] and
             curr["close"] > prev["open"] and curr["open"] < prev["close"]) or
            (prev["close"] > prev["open"] and curr["close"] < curr["open"] and
             curr["open"] > prev["close"] and curr["close"] < prev["open"])
        )

    def generate_signal(self, curr: dict, prev: dict) -> Optional[Signal]:
        if self.is_pattern(curr, prev):
            signal_type = SignalType.BUY if curr["close"] > curr["open"] else SignalType.SELL
            return Signal(self.symbol, curr["timestamp"], signal_type, strategy_name=self.__class__.__name__)
        return None

class MorningStarPattern(CandlePatternStrategy):
    def is_pattern(self, k1: dict, k2: dict, k3: dict) -> bool:
        return (
            k1["close"] < k1["open"] and
            abs(k2["close"] - k2["open"]) < 0.5 * abs(k1["close"] - k1["open"]) and
            k3["close"] > k3["open"] and k3["close"] > ((k1["open"] + k1["close"]) / 2)
        )

    def generate_signal(self, k3: dict, k2: dict, k1: dict) -> Optional[Signal]:
        if self.is_pattern(k1, k2, k3):
            return Signal(self.symbol, k3["timestamp"], SignalType.BUY, strategy_name=self.__class__.__name__)
        return None


class EveningStarPattern(CandlePatternStrategy):
    def is_pattern(self, k1: dict, k2: dict, k3: dict) -> bool:
        return (
            k1["close"] > k1["open"] and
            abs(k2["close"] - k2["open"]) < 0.5 * abs(k1["close"] - k1["open"]) and
            k3["close"] < k3["open"] and k3["close"] < ((k1["open"] + k1["close"]) / 2)
        )

    def generate_signal(self, k3: dict, k2: dict, k1: dict) -> Optional[Signal]:
        if self.is_pattern(k1, k2, k3):
            return Signal(self.symbol, k3["timestamp"], SignalType.SELL, strategy_name=self.__class__.__name__)
        return None

class PiercingPattern(CandlePatternStrategy):
    def is_pattern(self, k1: dict, k2: dict) -> bool:
        return (
            k1["close"] < k1["open"] and k2["open"] < k1["close"] and
            k2["close"] > ((k1["close"] + k1["open"]) / 2)
        )

    def generate_signal(self, k2: dict, k1: dict) -> Optional[Signal]:
        if self.is_pattern(k1, k2):
            return Signal(self.symbol, k2["timestamp"], SignalType.BUY, strategy_name=self.__class__.__name__)
        return None

class DarkCloudCoverPattern(CandlePatternStrategy):
    def is_pattern(self, k1: dict, k2: dict) -> bool:
        return (
            k1["close"] > k1["open"] and k2["open"] > k1["high"] and
            k2["close"] < ((k1["close"] + k1["open"]) / 2)
        )

    def generate_signal(self, k2: dict, k1: dict) -> Optional[Signal]:
        if self.is_pattern(k1, k2):
            return Signal(self.symbol, k2["timestamp"], SignalType.SELL, strategy_name=self.__class__.__name__)
        return None