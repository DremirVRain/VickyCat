import asyncio
import functools
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from collections import deque
from typing import Dict, Any, List
from longport.openapi import QuoteContext, Config, SubType, PushQuote, PushDepth, PushBrokers, PushTrades
from database import DatabaseManager
from config import symbols, TRADE_SESSION
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def to_utc_iso(dt: datetime) -> str:
    """将 datetime 保留为原始 UTC ISO 字符串（如果是 naive datetime，则假设为 UTC）"""
    if dt.tzinfo is None:
        # 假设 SDK 反序列化时丢失 tzinfo，我们强制假定它本来就是 UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace('+00:00', 'Z')

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
        self.trade_queue = {symbol: deque(maxlen=1000) for symbol in symbols}        
        self._quote_callback = None
        self._last_processed_minute = None

    def set_quote_callback(self, callback):
        self._quote_callback = callback

    def on_quote(self, symbol: str, quote: PushQuote):
        print(symbol, quote)
        """行情推送回调"""            
        
        # # 直接从trades_msg的原始表示中提取时间戳
        # # 假设trades_msg有原始响应文本
        # if hasattr(quote, '_raw_data'):
        #     # 解析_raw_data获取原始时间戳
        #     pass
        # else:
        #     # 使用正则从打印输出中提取
        #     quote_str = str(quote)
        #     utc_str = quote_str.split('timestamp: "')[1].split('"')[0]
        quote_data = {
            "timestamp":  quote.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "price": float(quote.last_done),
            "open": float(quote.open),
            "high": float(quote.high),
            "low": float(quote.low),
            "volume": int(quote.volume or 0),
            "turnover": float(quote.turnover or 0),
            "current_volume": int(quote.current_volume or 0),
            "current_turnover": float(quote.current_turnover or 0),
            "trade_status": str(quote.trade_status),
            "trade_session": str(quote.trade_session)
        }

        #self.db_manager.data_cache.update_cache(symbol, quote_data)
        self.quote_queue[symbol].append(quote_data)

        # if self._quote_callback:
        #     self._quote_callback(symbol, quote_data)

    def on_depth(self, symbol: str, depth: PushDepth):
        print(symbol, depth)

    def on_trades(self, symbol: str, trades_msg: PushTrades):
        print(symbol, trades_msg)
        """逐笔成交推送回调"""
        for trade in trades_msg.trades:        
            # # 直接从trades_msg的原始表示中提取时间戳
            # # 假设trades_msg有原始响应文本
            # if hasattr(trades_msg, '_raw_data'):
            #     # 解析_raw_data获取原始时间戳
            #     pass
            # else:
            #     # 使用正则从打印输出中提取
            #     trade_str = str(trade)
            #     utc_str = trade_str.split('timestamp: "')[1].split('"')[0]

            trade_data = {
                "timestamp": trade.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": symbol,
                "price": float(trade.price),
                "volume": int(trade.volume),
                "trade_type": str(trade.trade_type),
                "direction": str(trade.direction),
                "trade_session": str(trade.trade_session),
            }
            self.trade_queue[symbol].append(trade_data)

    @handle_exception
    async def start_subscription(self):
        """启动行情订阅"""
        self.ctx.set_on_quote(self.on_quote)
        self.ctx.set_on_depth(self.on_depth)
        self.ctx.set_on_trades(self.on_trades)
        self.ctx.subscribe(symbols, [SubType.Quote, SubType.Depth, SubType.Trade], True)
        logging.info("订阅行情成功")

    @handle_exception
    async def data_saver(self):
        """异步保存行情数据"""
        while True:
            quote_tasks = [self.save_quote_data(symbol) for symbol in symbols]
            trade_tasks = [self.save_trade_data(symbol) for symbol in symbols]
            await asyncio.gather(*(quote_tasks + trade_tasks))
            await asyncio.sleep(0.1)

    @handle_exception
    async def save_quote_data(self, symbol: str):
        """保存单个 symbol 的行情数据"""
        quotes = list(self.quote_queue[symbol])
        self.quote_queue[symbol].clear()
        if quotes:
            await self.db_manager.save_quotes_batch(quotes)

    @handle_exception
    async def save_trade_data(self, symbol: str):
        """保存逐笔成交数据"""
        trades = list(self.trade_queue[symbol])
        self.trade_queue[symbol].clear()
        if trades:
            await self.db_manager.save_trades_batch(trades)

    @handle_exception
    async def schedule_archive_task(self):
        """调度归档任务，每天盘后 16:10 触发"""
        while True:
            now = datetime.now()
            if now.hour == 16 and now.minute == 10:
                logging.info("开始数据归档...")
                self.db_manager.archive_old_data()
            await asyncio.sleep(60)

    # @handle_exception
    # async def minute_watcher(self):
    #     """每分钟检查是否有跳空未闭合蜡烛，补发空蜡烛或触发闭合"""
    #     while True:
    #         now = datetime.now()
    #         current_minute = now.replace(second=0, microsecond=0)

    #         if self._last_processed_minute and current_minute > self._last_processed_minute:
    #             for symbol in symbols:
    #                 last_min_str = self._last_processed_minute.strftime("%Y-%m-%d %H:%M")
    #                 cached_data = self.db_manager.data_cache.get_cached_data(
    #                     symbol,
    #                     start_time=last_min_str + ":00",
    #                     end_time=last_min_str + ":59"
    #                 )

    #                 if cached_data:
    #                     if self._kline_callback:
    #                         self._kline_callback(symbol, cached_data[-1])
    #                 else:
    #                     last_price = self.db_manager.get_last_price(symbol) or 0.0
    #                     empty_candle = {
    #                         "timestamp": self._last_processed_minute.strftime("%Y-%m-%d %H:%M:%S"),
    #                         "symbol": symbol,
    #                         "price": last_price,
    #                         "volume": 0,
    #                         "turnover": 0.0
    #                     }
    #                     self.db_manager.data_cache.update_cache(symbol, empty_candle)
    #                     await self.db_manager.save_quotes_batch([empty_candle])
    #                     if self._kline_callback:
    #                         self._kline_callback(symbol, empty_candle)
    #                     logging.warning(f"[{symbol}] 补发空蜡烛 for {last_min_str}")

    #         self._last_processed_minute = current_minute
    #         await asyncio.sleep(60 - now.second)

    async def run(self):
        """启动 DataFeed 模块"""
        await asyncio.gather(
            self.start_subscription(),
            self.data_saver(),
            self.schedule_archive_task(),
            #self.minute_watcher(),
        )
