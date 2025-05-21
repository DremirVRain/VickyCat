# signal.py
from enum import Enum
from datetime import datetime
from typing import Optional


class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class Signal:
    def __init__(
        self,
        symbol: str,
        timestamp: datetime,
        signal_type: SignalType,
        strength: float = 1.0,
        strategy_name: Optional[str] = None,
        metadata: Optional[dict] = None
    ):
        self.symbol = symbol
        self.timestamp = timestamp
        self.signal_type = signal_type
        self.strength = strength
        self.strategy_name = strategy_name
        self.metadata = metadata or {}

    def __repr__(self):
        return f"<Signal {self.signal_type.value.upper()} | {self.symbol} | {self.timestamp} | strength={self.strength:.2f} | from={self.strategy_name}>"
