# strategies/base_strategy.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from datetime import datetime
from strategy.strategy_signal import Signal, SignalType
from dataclasses import dataclass

@dataclass
class MarketContext:
    is_uptrend: bool = False
    near_support: bool = False
    near_resistance: bool = False
    volatility: float = 0.0
    trend_strength: float = 0.0

    ma_short: Optional[float] = None  # 短期均线，如5日/5分钟
    ma_mid: Optional[float] = None    # 中期均线，如20日/20分钟
    ma_long: Optional[float] = None   # 长期均线，如60日/60分钟
    recent_high: Optional[float] = None  # 最近N周期高点
    recent_low: Optional[float] = None   # 最近N周期低点
    volume_avg: Optional[float] = None   # 平均成交量（N周期）
    rsi: Optional[float] = None          # RSI 指标值
    atr: Optional[float] = None          # ATR 波动率指标    
    
    micro_prices: Optional[List[float]] = None        # 过去60s收盘价（或最新价）
    micro_volumes: Optional[List[float]] = None       # 过去60s成交量
    micro_turnover: Optional[List[float]] = None      # 过去60s成交额
    micro_max_price: Optional[float] = None           # 子分钟高
    micro_min_price: Optional[float] = None           # 子分钟低
    tick_count: Optional[int] = None                  # 1s数据点数量

    # 滑动K线窗口
    recent_klines: Optional[List[Dict]] = None

    extra: Dict[str, Any] = {}


class BaseStrategy(ABC):
    def __init__(self, symbol: str, window_size: int = 10,  debug: bool = False):
        self.symbol = symbol
        self.debug = debug

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
            price=kline["close"],
            strength=strength,
            strategy_name=self.__class__.__name__,
            metadata=metadata or {}
        )

    def __str__(self):
        return f"{self.__class__.__name__}(symbol={self.symbol})"

    def __repr__(self):
        return self.__str__()


class BuySellStrategy(BaseStrategy):
    """返回 BUY/SELL 类型信号的策略"""
    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        pass  # 子类实现具体逻辑


class TrendStrategy(BaseStrategy):
    """返回 TREND_UP/TREND_DOWN 类型信号的策略"""
    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        pass  # 子类实现具体逻辑


class CompositeStrategy(BaseStrategy):
    def __init__(self, symbol: str, strategies: List[BaseStrategy], debug: bool = False):
        super().__init__(symbol, debug)
        self.strategies = strategies

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        all_signals = []
        for strategy in self.strategies:
            sig = strategy.generate_signal(context=context)
            if sig:
                self.log(f"子策略 {strategy} 产生信号：{sig}")
                all_signals.append(sig)

        # 默认返回第一个 BUY/SELL 信号，趋势类信号不直接输出
        for sig in all_signals:
            if sig.signal_type in {SignalType.BUY, SignalType.SELL}:
                return sig

        return None
