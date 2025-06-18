# strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from strategy.strategy_signal import Signal, SignalType, create_signal  
from dataclasses import dataclass, field

@dataclass
class MarketContext:
    trend_info: Optional[Dict[str, Union[str, float]]] = None  # {"direction": "up", "strength": 0.73}
    
    volatility: float = 0.0
    ma_short: Optional[float] = None
    ma_mid: Optional[float] = None
    ma_long: Optional[float] = None
    recent_high: Optional[float] = None
    recent_low: Optional[float] = None
    volume_avg: Optional[float] = None

    micro_prices: List[float] = field(default_factory=list)
    micro_volumes: List[float] = field(default_factory=list)
    micro_turnover: List[float] = field(default_factory=list)
    micro_max_price: Optional[float] = None
    micro_min_price: Optional[float] = None
    tick_count: int = 0

    recent_klines: List[dict] = field(default_factory=list)  # 供策略回看结构
    
    @property
    def is_uptrend(self) -> Optional[bool]:
        if self.trend_info and "direction" in self.trend_info:
            return self.trend_info["direction"] == "up"
        return None

    @property
    def trend_strength(self) -> float:
        if self.trend_info and "strength" in self.trend_info:
            return self.trend_info["strength"]
        return 0.0


class BaseStrategy(ABC):
    strategy_category: str = "generic"

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.params = {}

    def log(self, msg: str):
        if self.debug:
            print(f"[{self.__class__.__name__}] {msg}")

    @classmethod
    def required_klines(cls) -> int:
        """策略所需的最小K线数量（滑动窗口大小），主控模块根据它准备 context.recent_klines"""
        return 3  # 默认需要最近3根K线

    @abstractmethod
    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        """生成交易信号，支持传入市场环境信息"""
        pass

    def build_signal(
        self,
        kline: dict,
        signal_type: SignalType,
        strength: float = 1.0,
        metadata: Optional[dict] = None,
        expected_action: Optional[str] = None
    ) -> Signal:
        """
        统一构建 Signal 对象，自动注入策略名称、时间戳、收盘价、预期操作等。
        """
        ts = kline.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")

        return create_signal(
            symbol=self.symbol,
            signal_type=signal_type,
            strategy_name=self.__class__.__name__,
            strength=strength,
            timestamp=ts,
            price=kline["close"],
            metadata=metadata,
            expected_action=expected_action
        )

    def __str__(self):
        return f"{self.__class__.__name__}(symbol={self.symbol})"

    def __repr__(self):
        return self.__str__()

