from typing import Callable, Dict, List
from strategy.base_strategy import BaseStrategy
from strategy.strategy_signal import Signal
from datetime import datetime
from database import DatabaseManager

class StrategyManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager  # 直接挂接 DatabaseManager
        self.strategies: Dict[str, List[BaseStrategy]] = {}  # symbol -> strategies
        self.signal_callback: Callable[[str, Signal], None] = None

    def register_strategy(self, symbol: str, strategy: BaseStrategy):
        if symbol not in self.strategies:
            self.strategies[symbol] = []
        self.strategies[symbol].append(strategy)

    def set_signal_callback(self, callback: Callable[[str, Signal], None]):
        self.signal_callback = callback

    def on_kline(self, symbol: str, kline: Dict):
        # 获取 1s 子分钟数据作为附加分析
        end_time = kline["timestamp"]
        start_time = self._get_minute_start(end_time)
        data_1s = self.db_manager.get_kline_1s(start_time, end_time, symbol)

        for strategy in self.strategies.get(symbol, []):
            signal = strategy.generate_signal(kline, data_1s)
            if signal and self.signal_callback:
                self.signal_callback(symbol, signal)

    def _get_minute_start(self, timestamp: str) -> str:
        # 将 timestamp 转为所在分钟的起始时间字符串
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        dt_minute_start = dt.replace(second=0)
        return dt_minute_start.strftime("%Y-%m-%d %H:%M:%S")
