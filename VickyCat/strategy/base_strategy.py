# strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
from strategy.strategy_signal import Signal, SignalType


class BaseStrategy(ABC):
    def __init__(self, symbol: str):
        self.symbol = symbol

    @abstractmethod
    def generate_signal(self, *klines: dict) -> Optional[Signal]:
        pass

    def build_signal(
        self,
        kline: dict,
        signal_type: SignalType,
        strength: float = 1.0,
        metadata: Optional[dict] = None
    ) -> Signal:
        """统一构建 Signal 对象，自动注入策略名称、时间戳等。"""
        ts = kline.get("timestamp")
        # 支持字符串或 datetime 对象
        if isinstance(ts, str):
            ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return Signal(
            symbol=self.symbol,
            timestamp=ts,
            signal_type=signal_type,
            strength=strength,
            strategy_name=self.__class__.__name__,
            metadata=metadata or {}
        )

    def __str__(self):
        return f"{self.__class__.__name__}(symbol={self.symbol})"

    def __repr__(self):
        return self.__str__()
