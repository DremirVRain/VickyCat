import asyncio
import logging
from datetime import datetime
from typing import Optional, List
from database import DatabaseManager
from trade import Trade
from data_feed import DataFeed
from config import symbols

class Backtester:
    def __init__(self, symbol_list: Optional[List[str]] = None, start_time: Optional[str] = None, end_time: Optional[str] = None, debug_mode: bool = False):
        self.symbol_list = symbol_list if symbol_list else symbols
        self.start_time = start_time  # 格式："2024-01-01 09:30:00"
        self.end_time = end_time      # 格式："2024-01-01 12:00:00"
        self.debug_mode = debug_mode

        self.db = DatabaseManager()
        self.trade = Trade()
        self.data_feed = DataFeed()  # 如果你希望策略用 on_quote，可从此类中引入

        # 模拟时间，在 debug 模式下作为 datetime.now() 替代
        self._mock_time = None

    def get_now(self) -> datetime:
        """替代 datetime.now() 的统一接口"""
        return self._mock_time if self.debug_mode else datetime.now()

    async def run(self):
        logging.info(f"启动回测，标的: {self.symbol_list}")
        for symbol in self.symbol_list:
            logging.info(f"读取数据: {symbol}")
            data = self.db.get_quotes(symbol, self.start_time, self.end_time)
            logging.info(f"数据条数: {len(data)}")

            for idx, quote in enumerate(data):
                # 设置模拟当前时间
                if self.debug_mode:
                    self._mock_time = datetime.strptime(quote['timestamp'], "%Y-%m-%d %H:%M:%S")

                # 模拟推送给 on_quote（策略接入点）
                self.data_feed.on_quote(symbol, quote)

                # 你可以在这里判断订单状态、模拟成交等
                if idx + 1 < len(data):
                    next_quote = data[idx + 1]
                    # 可在此处模拟订单撮合等行为

                await asyncio.sleep(0)  # 释放事件循环

        logging.info("回测结束")

# 如果直接运行
if __name__ == "__main__":
    bt = Backtester(debug_mode=True)
    asyncio.run(bt.run())

