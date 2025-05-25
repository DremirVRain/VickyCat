import asyncio
import functools
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from collections import deque
from typing import Dict, Any, List
from longport.openapi import QuoteContext, Config, SubType, PushQuote
from database import DatabaseManager
from config import symbols, TRADE_SESSION

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def handle_exception(func):
    """统一异步异常处理装饰器"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"[{func.__name__}] Exception: {e}")
    return wrapper

class DataFeed:
    def __init__(self):
        self.ctx = QuoteContext(Config.from_env())
        self.db_manager = DatabaseManager()
        self.quote_queue = {symbol: deque(maxlen=1000) for symbol in symbols}       
        self._kline_callback = None

    def set_kline_callback(self, callback):
        """设置闭合1分钟K线数据回调（来自数据库或内部生成）"""
        self._kline_callback = callback

    def on_quote(self, symbol: str, quote: PushQuote):
        """行情推送回调"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        quote_data = {
            "timestamp": timestamp,
            "symbol": symbol,
            "price": float(quote.last_done),
            "volume": int(quote.current_volume or 0),
            "turnover": float(quote.current_turnover or 0.0)
        }
        
        current_minute = timestamp[:16]
        # 检查 quote_queue 是否有上一个 quote
        q = self.quote_queue[symbol]
        if q:
            last_minute = q[-1]["timestamp"][:16]
            if last_minute != current_minute:
                # 旧分钟闭合，生成闭合蜡烛
                closed_candles = self.db_manager.data_cache.get_cached_data(
                    symbol,
                    start_time=last_minute + ":00",
                    end_time=last_minute + ":59"
                )
                if closed_candles and self._kline_callback:
                    self._kline_callback(symbol, closed_candles[-1])
                else:
                    print(f"[{symbol}] No closed kline found for {last_minute}")

        # 更新缓存，replace=True 表示更新未闭合的蜡烛数据
        self.db_manager.data_cache.update_cache(symbol, quote_data)
        
        # 插入数据库并更新缓存
        self.quote_queue[symbol].append(quote_data)

    @handle_exception
    async def start_subscription(self):
        """启动行情订阅"""
        self.ctx.set_on_quote(self.on_quote)
        self.ctx.subscribe(symbols, [SubType.Quote], True)
        logging.info("订阅行情成功")

    @handle_exception
    async def data_saver(self):
        """异步保存行情数据"""
        while True:
            tasks = [self.save_quote_data(symbol) for symbol in symbols]
            if tasks:
                await asyncio.gather(*tasks)

            await asyncio.sleep(0.1)

    @handle_exception
    async def save_quote_data(self, symbol: str):
        """保存单个 symbol 的行情数据"""
        quotes = list(self.quote_queue[symbol])
        self.quote_queue[symbol].clear()
        if quotes:
            await self.db_manager.save_quotes_batch(quotes)

    @handle_exception
    async def schedule_archive_task(self):
        """调度归档任务，每天盘后 16:10 触发"""
        while True:
            now = datetime.now()
            if now.hour == 16 and now.minute == 10:
                logging.info("开始数据归档...")
                self.db_manager.archive_old_data()

            await asyncio.sleep(60)

    async def run(self):
        """启动 DataFeed 模块"""
        await asyncio.gather(
            self.start_subscription(),
            self.data_saver(),
            self.schedule_archive_task(),
        )
