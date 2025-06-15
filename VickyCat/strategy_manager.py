from typing import Callable, Dict, List, Optional, Any
from strategy.base_strategy import BaseStrategy, MarketContext
from strategy.strategy_signal import Signal, SignalType
from datetime import datetime, timedelta
from database import DatabaseManager
from collections import defaultdict, deque
from strategy.strategy_utils import detect_trend

STRATEGY_WEIGHTS = {
    "pattern": 0.3,
    "indicator": 0.2,
    "microstructure": 0.4,
    "structure": 0.1
}

class StrategyFusionManager:
    def __init__(self, strategy_weights: Dict[str, float]):
        self.strategy_weights = strategy_weights
        self.signals_by_symbol: Dict[str, List[Signal]] = defaultdict(list)

    def collect_signal(self, symbol: str, signal: Signal, strategy: BaseStrategy):
        """收集每个策略输出的信号"""
        signal.metadata["strategy_category"] = strategy.strategy_category
        self.signals_by_symbol[symbol].append(signal)

    def clear(self, symbol: str):
        self.signals_by_symbol[symbol].clear()

    def fuse_signals(self, symbol: str) -> Optional[Signal]:
        """融合策略生成的信号（按类型加权）"""
        signals = self.signals_by_symbol[symbol]
        if not signals:
            return None

        buy_strength = 0.0
        sell_strength = 0.0
        total_weight = 0.0

        for sig in signals:
            category = sig.metadata.get("strategy_category", "generic")
            weight = self.strategy_weights.get(category, 0.0)
            if sig.signal_type == SignalType.BUY:
                buy_strength += sig.strength * weight
            elif sig.signal_type == SignalType.SELL:
                sell_strength += sig.strength * weight
            total_weight += weight

        if buy_strength > sell_strength and buy_strength > 0.1:
            return Signal(
                symbol=symbol,
                timestamp=signals[-1].timestamp,
                signal_type=SignalType.BUY,
                price=signals[-1].price,
                strength=buy_strength / total_weight,
                strategy_name="FusionManager"
            )
        elif sell_strength > buy_strength and sell_strength > 0.1:
            return Signal(
                symbol=symbol,
                timestamp=signals[-1].timestamp,
                signal_type=SignalType.SELL,
                price=signals[-1].price,
                strength=sell_strength / total_weight,
                strategy_name="FusionManager"
            )
        return None


class StrategyManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager  # 直接挂接 DatabaseManager
        self.strategies: Dict[str, List[BaseStrategy]] = {}  # symbol -> strategies
        self.signal_callback: Callable[[str, Signal], None] = None  # 用户自定义处理 signal 的回调
        self.fusion_manager = StrategyFusionManager(STRATEGY_WEIGHTS)
        self.current_minute_kline: Dict[str, Dict] = {}

    def register_strategy(self, symbol: str, strategy: BaseStrategy):
        """注册策略到指定 symbol"""
        if symbol not in self.strategies:
            self.strategies[symbol] = []
        self.strategies[symbol].append(strategy)

    def set_signal_callback(self, callback: Callable[[str, Signal], None]):
        """设置 signal 生成后的回调函数，常用于记录或下单"""
        self.signal_callback = callback

    def on_quote(self, symbol: str, quote: Dict[str, Any], index=None):
        self.kline_windows[symbol].append(quote)  # 用 quote 模拟小时间窗
        context = self._build_market_context(symbol, quote["timestamp"])
        context.recent_quotes = list(self.kline_windows[symbol])  # 注意：已非 kline，而是 quote

        self.fusion_manager.clear(symbol)
        for strategy in self.strategies.get(symbol, []):
            signal = strategy.generate_signal(quote, context=context)
            if signal:
                self.fusion_manager.collect_signal(symbol, signal, strategy)

        fused_signal = self.fusion_manager.fuse_signals(symbol)
        if fused_signal and self.signal_callback:
            self.signal_callback(symbol, fused_signal, index)

    def _get_minute_start(self, timestamp: str) -> str:
        """将 timestamp 转为所在分钟的起始时间字符串"""
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        dt_minute_start = dt.replace(second=0)
        return dt_minute_start.strftime("%Y-%m-%d %H:%M:%S")

    def _build_market_context(self, symbol: str, end_time_str: str) -> MarketContext:
        end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
        start_time = end_time - timedelta(minutes=100)
        rows = self.db_manager.get_kline_1m(symbol, start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time_str)

        if not rows:
            return MarketContext()  # 返回空默认值

        closes = [r["close"] for r in rows if r["close"] is not None]
        highs = [r["high"] for r in rows if r["high"] is not None]
        lows = [r["low"] for r in rows if r["low"] is not None]
        volumes = [r["volume"] for r in rows if r["volume"] is not None]

        # 移动均线
        def sma(data: List[float], n: int) -> Optional[float]:
            if len(data) >= n:
                return sum(data[-n:]) / n
            return None

        ma_short = sma(closes, 5)
        ma_mid = sma(closes, 20)
        ma_long = sma(closes, 60)

        # 趋势识别增强
        trend_info = detect_trend(rows, window=5) or {"direction": "sideways", "strength": 0.0}
        is_uptrend = trend_info["direction"] == "up"
        trend_strength = trend_info["strength"]

        # 震荡度/波动性
        volatility = sma([abs(h - l) for h, l in zip(highs, lows)], 20)
        recent_high = max(highs[-20:]) if len(highs) >= 20 else None
        recent_low = min(lows[-20:]) if len(lows) >= 20 else None
        volume_avg = sma(volumes, 20)

        # 微结构数据
        micro_data = list(self.db_manager.data_cache.cache.get(symbol, []))[-60:]
        micro_prices = [x["price"] for x in micro_data if "price" in x]
        micro_volumes = [x["volume"] for x in micro_data if "volume" in x]
        micro_turnovers = [x["turnover"] for x in micro_data if "turnover" in x]
        micro_max = max(micro_prices) if micro_prices else None
        micro_min = min(micro_prices) if micro_prices else None
        tick_count = len(micro_data)

        return MarketContext(
            volatility=volatility or 0.0,
            ma_short=ma_short,
            ma_mid=ma_mid,
            ma_long=ma_long,
            recent_high=recent_high,
            recent_low=recent_low,
            volume_avg=volume_avg,
            trend_info=trend_info,  # 新增字段，结构为 {"direction": "up", "strength": 0.7}

            micro_prices=micro_prices,
            micro_volumes=micro_volumes,
            micro_turnover=micro_turnovers,
            micro_max_price=micro_max,
            micro_min_price=micro_min,
            tick_count=tick_count
        )