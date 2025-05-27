# backtester.py

from strategy_manager import StrategyManager
from strategy.strategy_signal import Signal
from strategy.candle_pattern_strategy import HammerPattern  # 示例策略
from database import DatabaseManager
from datetime import datetime
import time
from utils.time_util import convert_to_eastern

class Backtester:
    def __init__(self, db_path: str, symbols: list):
        self.db_manager = DatabaseManager()
        self.symbols = symbols
        self.strategy_manager = StrategyManager(self.db_manager)
        self.results = []  # 存储所有 signal，可后续分析或输出

    def setup_strategies(self):
        # 这里注册你要测试的策略，可以灵活替换和扩展
        for symbol in self.symbols:
            self.strategy_manager.register_strategy(symbol, HammerPattern(symbol))

        # 设置 signal 回调
        self.strategy_manager.set_signal_callback(self.on_signal)

    def on_signal(self, symbol: str, signal: Signal):
        print(f"[{convert_to_eastern(signal.timestamp)}] {symbol} 触发信号: {signal.signal_type}")
        self.results.append((symbol, signal))

    def run(self, start_date: str = None, end_date: str = None):
        self.setup_strategies()

        for symbol in self.symbols:
            print(f"🚀 开始回测 {symbol} 数据")
            klines = self.db_manager.get_kline_1m(symbol, start_date, end_date)
            print(f"共加载 {len(klines)} 根分钟K线")

            for kline in klines:
                self.strategy_manager.on_kline(symbol, kline)
                time.sleep(0.001)  # 模拟推送节奏，可关闭以提速

        print(f"✅ 回测完成，共触发 {len(self.results)} 个信号")


if __name__ == "__main__":
    symbols = ["TSLA.US"]
    db_path = "minute_data.db"
    start_date = "2025-05-19 21:30:00"
    end_date = "2025-05-20 05:30:00"

    backtester = Backtester(db_path, symbols)
    backtester.run(start_date, end_date)
