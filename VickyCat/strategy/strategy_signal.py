# strategy_signal.py
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any

class SignalType(Enum):
    BUY = "buy"              # 买入信号（看涨）
    SELL = "sell"            # 卖出信号（看跌）
    HOLD = "hold"            # 持有，无操作
    # TREND_UP = "trend_up"    # 趋势向上（趋势确认）#这两个没有策略输出，暂时注释掉
    # TREND_DOWN = "trend_down" # 趋势向下（趋势确认）
    FILTER = "filter"      # 过滤信号，仅用于辅助其他信号判断
    INDICATOR = "indicator"  # 指标信号，基于技术指标的信号
    STRUCTURE = "structure"  # 结构信号，基于价格结构的信号


class Signal:
    def __init__(
        self,
        symbol: str,
        timestamp: datetime,
        signal_type: SignalType,
        price: float,
        strength: float = 1.0,
        strategy_name: Optional[str] = None,
        metadata: Optional[dict] = None,
        expected_action: Optional[str] = None
    ):
        self.symbol = symbol
        self.timestamp = timestamp
        self.signal_type = signal_type
        self.price = price
        self.strength = strength
        self.strategy_name = strategy_name
        self.metadata = metadata or {}
        self.expected_action = expected_action or {}

    def __repr__(self):
        return f"<Signal {self.signal_type.value.upper()} | {self.symbol} | {self.timestamp} | strength={self.strength:.2f} | from={self.strategy_name}>"

def create_signal(
    symbol: str,
    signal_type: SignalType,
    strategy_name: str,
    strength: float,
    timestamp: Optional[Any] = None,
    price: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    expected_action: Optional[str] = None
) -> Signal:
    """
    统一创建 Signal 实例，确保必需字段齐全，避免遗漏。

    Args:
        symbol: 标的代码
        signal_type: 信号类型 (SignalType枚举)
        strategy_name: 策略类名称
        strength: 信号强度
        timestamp: 时间戳，默认取当前时间
        price: 当前价格，默认0
        metadata: 附加信息字典

    Returns:
        Signal 实例
    """
    if timestamp is None:
        timestamp = datetime.now()
    if price is None:
        price = 0.0
    if metadata is None:
        metadata = {}

    return Signal(
        symbol=symbol,
        timestamp=timestamp,
        price=price,
        signal_type=signal_type,
        strategy_name=strategy_name,
        strength=strength,
        metadata=metadata,
        expected_action=expected_action
    )