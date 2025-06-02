from typing import Callable, Dict, List, Optional
from strategy.base_strategy import BaseStrategy, MarketContext
from strategy.strategy_signal import Signal
from datetime import datetime
from database import DatabaseManager

class StrategyManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager  # 直接挂接 DatabaseManager
        self.strategies: Dict[str, List[BaseStrategy]] = {}  # symbol -> strategies
        self.signal_callback: Callable[[str, Signal], None] = None  # 用户自定义处理 signal 的回调

    def register_strategy(self, symbol: str, strategy: BaseStrategy):
        """注册策略到指定 symbol"""
        if symbol not in self.strategies:
            self.strategies[symbol] = []
        self.strategies[symbol].append(strategy)

    def set_signal_callback(self, callback: Callable[[str, Signal], None]):
        """设置 signal 生成后的回调函数，常用于记录或下单"""
        self.signal_callback = callback

    def on_kline(self, symbol: str, kline: Dict, index=None):
        """
        主入口：接收一根 1 分钟级别的 K 线，触发所有该 symbol 下已注册策略。
        会自动拉取该分钟内的 1 秒数据，供策略分析。
        """
        end_time = kline["timestamp"]
        start_time = self._get_minute_start(end_time)
        #data_1s = self.db_manager.get_kline_1s(symbol, start_time, end_time)
        
        # 构建市场上下文
        context = self._build_market_context(symbol, end_time)

        for strategy in self.strategies.get(symbol, []):
            signal = strategy.generate_signal(kline, context=context)
            if signal and self.signal_callback:
                self.signal_callback(symbol, signal, index)

    def _get_minute_start(self, timestamp: str) -> str:
        """将 timestamp 转为所在分钟的起始时间字符串"""
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        dt_minute_start = dt.replace(second=0)
        return dt_minute_start.strftime("%Y-%m-%d %H:%M:%S")

    def _build_market_context(self, symbol: str, end_time_str: str) -> MarketContext:
        """构造 MarketContext 对象，用于传入策略"""
        end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
        start_time = end_time - timedelta(minutes=100)  # 拉取过去100分钟K线
        rows = self.db_manager.get_kline_1m(symbol, start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time_str)

        closes = [r["close"] for r in rows if r["close"] is not None]
        highs = [r["high"] for r in rows if r["high"] is not None]
        lows = [r["low"] for r in rows if r["low"] is not None]
        volumes = [r["volume"] for r in rows if r["volume"] is not None]

        def sma(data: List[float], n: int) -> Optional[float]:
            if len(data) >= n:
                return sum(data[-n:]) / n
            return None

        ma_short = sma(closes, 5)
        ma_mid = sma(closes, 20)
        ma_long = sma(closes, 60)

        is_uptrend = ma_short and ma_long and ma_short > ma_long
        trend_strength = ((ma_short - ma_long) / ma_long) if (ma_short and ma_long and ma_long != 0) else 0

        recent_high = max(highs[-20:]) if len(highs) >= 20 else None
        recent_low = min(lows[-20:]) if len(lows) >= 20 else None
        volume_avg = sma(volumes, 20)
        volatility = sma([abs(h - l) for h, l in zip(highs, lows)], 20)

        # 构造并返回 MarketContext 实例
        return MarketContext(
            is_uptrend=is_uptrend,
            trend_strength=trend_strength,
            volatility=volatility or 0.0,
            ma_short=ma_short,
            ma_mid=ma_mid,
            ma_long=ma_long,
            recent_high=recent_high,
            recent_low=recent_low,
            volume_avg=volume_avg
        )