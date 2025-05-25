# strategy_signal.py
from enum import Enum
from datetime import datetime
from typing import Optional

class SignalType(Enum):
    BUY = "buy"              # 买入信号（看涨）
    SELL = "sell"            # 卖出信号（看跌）
    HOLD = "hold"            # 持有，无操作
    TREND_UP = "trend_up"    # 趋势向上（趋势确认）
    TREND_DOWN = "trend_down" # 趋势向下（趋势确认）
    WARNING = "warning"      # 预警信号，可能反转或形态待确认
    FILTER = "filter"      # 过滤信号，仅用于辅助其他信号判断
    INDICATOR = "indicator"  # 指标信号，基于技术指标的信号
    STRUCTURE = "structure"  # 结构信号，基于价格结构的信号


class Signal:
    def __init__(
        self,
        symbol: str,
        timestamp: datetime,
        signal_type: SignalType,
        strength: float = 1.0,
        strategy_name: Optional[str] = None,
        metadata: Optional[dict] = None,
        expected_action: Optional[str] = None
    ):
        self.symbol = symbol
        self.timestamp = timestamp
        self.signal_type = signal_type
        self.strength = strength
        self.strategy_name = strategy_name
        self.metadata = metadata or {}
        self.expected_action = expected_action or {}

    def __repr__(self):
        return f"<Signal {self.signal_type.value.upper()} | {self.symbol} | {self.timestamp} | strength={self.strength:.2f} | from={self.strategy_name}>"

# class MarketCondition(Enum):
#     UPTREND = auto()
#     DOWNTREND = auto()
#     SIDEWAYS = auto()

# def determine_trend(klines: list[dict]) -> MarketCondition:
#     closes = [k["close"] for k in klines[-5:]]
#     if all(x < y for x, y in zip(closes, closes[1:])):
#         return MarketCondition.UPTREND
#     elif all(x > y for x, y in zip(closes, closes[1:])):
#         return MarketCondition.DOWNTREND
#     return MarketCondition.SIDEWAYS