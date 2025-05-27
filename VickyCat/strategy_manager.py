from typing import Callable, Dict, List
from strategy.base_strategy import BaseStrategy
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

    def on_kline(self, symbol: str, kline: Dict):
        """
        主入口：接收一根 1 分钟级别的 K 线，触发所有该 symbol 下已注册策略。
        会自动拉取该分钟内的 1 秒数据，供策略分析。
        """
        end_time = kline["timestamp"]
        start_time = self._get_minute_start(end_time)
        #data_1s = self.db_manager.get_kline_1s(symbol, start_time, end_time)

        for strategy in self.strategies.get(symbol, []):
            signal = strategy.generate_signal(kline)
            if signal and self.signal_callback:
                self.signal_callback(symbol, signal)

    def _get_minute_start(self, timestamp: str) -> str:
        """将 timestamp 转为所在分钟的起始时间字符串"""
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        dt_minute_start = dt.replace(second=0)
        return dt_minute_start.strftime("%Y-%m-%d %H:%M:%S")
