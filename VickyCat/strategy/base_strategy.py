# strategies/base_strategy.py
from abc import ABC, abstractmethod
from signal import Signal
from typing import Optional

class BaseStrategy:
    def __init__(self, symbol: str):
        self.symbol = symbol

    def generate_signal(self, kline: dict) -> Optional[Signal]:
        raise NotImplementedError