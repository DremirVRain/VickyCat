import os
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from config import TRADE_SESSION
from utils.archive_manager import ArchiveManager
from utils.data_cache import DataCache
from utils.trading_time_manager import TradingTimeManager
from collections import defaultdict

DB_PATH = "market_data_async.db"

class DatabaseManager:
    def __init__(self, db_path: str = DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.create_tables()

        # 初始化 ArchiveManager
        self.archive_manager = ArchiveManager()

        # 初始化缓存
        self.data_cache = DataCache()

    def create_tables(self):
        """初始化 quotes 表结构并设置复合主键"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quotes (
                timestamp TEXT,
                symbol TEXT,
                sequence INTEGER,
                price REAL,
                volume INTEGER,
                turnover REAL,
                PRIMARY KEY (timestamp, symbol, sequence)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quote_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                price REAL,
                volume INTEGER,
                turnover REAL,
                error_message TEXT
            )
        ''')
        self.conn.commit()

    def is_trading_session(exchange: str = 'US') -> bool:
        """
        检查当前是否为指定交易所的交易时间段内
        :param exchange: 交易所名称 ('US', 'HK', 'CN')
        :return: True 表示当前为交易时间段内，False 表示非交易时间段
        """
        manager = TradingTimeManager(exchange)

        return manager.is_trading_time()

    def get_next_sequence(self, timestamp: str, symbol: str) -> int:
        """获取当前秒内的下一个 sequence 值"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MAX(sequence) FROM quotes WHERE timestamp = ? AND symbol = ?
        ''', (timestamp, symbol))
        max_sequence = cursor.fetchone()[0]
        return (max_sequence + 1) if max_sequence is not None else 1

    async def save_quotes_batch(self, quote_data_list: List[Dict[str, Any]]):
        if not quote_data_list:
            return

        try:
            cursor = self.conn.cursor()
            data_to_insert = []
            error_data = []

            # 新增：记录每个 timestamp+symbol 的 sequence
            sequence_map = {}

            for data in quote_data_list:
                try:
                    symbol = data["symbol"]
                    timestamp = data["timestamp"]

                    key = (timestamp, symbol)
                    if key not in sequence_map:
                        # 先查数据库中已有最大值
                        sequence_map[key] = self.get_next_sequence(timestamp, symbol)
                    else:
                        sequence_map[key] += 1

                    seq = sequence_map[key]

                    print(f"[Quote] 插入: {timestamp} | {symbol} | {seq} | "
                          f"price={data['price']} vol={data['volume']} turnover={data['turnover']}")

                    data_to_insert.append((
                        timestamp, symbol, seq,
                        data['price'], data['volume'], data['turnover']
                    ))

                except KeyError as e:
                    print(f"[Warning] 数据字段缺失: {data} - {e}")
                    error_data.append((data, str(e)))

            if data_to_insert:
                cursor.executemany('''
                    INSERT INTO quotes (timestamp, symbol, sequence, price, volume, turnover)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', data_to_insert)
                self.conn.commit()

            for data, msg in error_data:
                self.log_error_data(data, msg)

        except Exception as e:
            print(f"[Error] 批量插入数据失败: {e}")
            self.conn.rollback()


    def log_error_data(self, data: Dict[str, Any], error_message: str):
        """记录异常数据到 quote_errors 表"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO quote_errors (timestamp, symbol, price, volume, turnover, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data.get('timestamp', "N/A"),
                data.get('symbol', "N/A"),
                data.get('price', 0.0),
                data.get('volume', 0),
                data.get('turnover', 0.0),
                error_message
            ))
            self.conn.commit()

        except sqlite3.Error as e:
            print(f"[Error] 记录异常数据失败: {e}")
            self.conn.rollback()

    def archive_old_data(self):
        """归档并删除旧数据，通过 ArchiveManager 统一管理"""
        self.archive_manager.archive_old_data()    
        
    def get_kline_1s(self, symbol: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """获取 1 秒 K 线数据，优先从缓存获取。"""
        cached_data = self.data_cache.get_cached_data(symbol, start_time, end_time)

        if cached_data:
            return cached_data

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT timestamp, price, volume, turnover
            FROM quotes
            WHERE timestamp BETWEEN ? AND ? AND symbol = ?
            ORDER BY timestamp ASC
        ''', (start_time, end_time, symbol))

        # 临时按秒分组数据，key是秒时间字符串，value是list of ticks
        ticks_per_second = defaultdict(list)

        for row in cursor.fetchall():
            ts_str, price, volume, turnover = row
            # 确保 timestamp 格式是 "%Y-%m-%d %H:%M:%S" 或类似的
            # 只取到秒，去除毫秒
            if isinstance(ts_str, str):
                dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            else:
                dt = ts_str
            second_key = dt.strftime("%Y-%m-%d %H:%M:%S")
            ticks_per_second[second_key].append({
                "price": price,
                "volume": volume,
                "turnover": turnover
            })

        candles = []
        for second_key in sorted(ticks_per_second.keys()):
            ticks = ticks_per_second[second_key]
            open_price = ticks[0]["price"]
            close_price = ticks[-1]["price"]
            high_price = max(t["price"] for t in ticks)
            low_price = min(t["price"] for t in ticks)
            total_volume = sum(t["volume"] for t in ticks)
            total_turnover = sum(t["turnover"] for t in ticks)

            candle = {
                "timestamp": second_key,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": total_volume,
                "turnover": total_turnover
            }
            candles.append(candle)

        return candles

    def get_kline_5s(self, symbol: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """基于 1s 数据生成 5s 数据"""
        data_1s = self.get_kline_1s(symbol, start_time, end_time)

        if not data_1s:
            return []

        kline_5s = []
        for i in range(0, len(data_1s), 5):
            segment = data_1s[i:i + 5]

            open_price = segment[0]["open"]
            close_price = segment[-1]["close"]
            high_price = max(item["high"] for item in segment)
            low_price = min(item["low"] for item in segment)
            volume = sum(item["volume"] for item in segment)
            turnover = sum(item["turnover"] for item in segment)

            kline_5s.append({
                "timestamp": segment[-1]["timestamp"],
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "turnover": turnover
            })

        return kline_5s

    def get_kline_1m(self, symbol: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """基于 5s 数据生成 1m 数据"""
        data_5s = self.get_kline_5s(symbol, start_time, end_time)

        if not data_5s:
            return []

        kline_1m = []
        for i in range(0, len(data_5s), 12):
            segment = data_5s[i:i + 12]

            open_price = segment[0]["open"]
            close_price = segment[-1]["close"]
            high_price = max(item["high"] for item in segment)
            low_price = min(item["low"] for item in segment)
            volume = sum(item["volume"] for item in segment)
            turnover = sum(item["turnover"] for item in segment)

            kline_1m.append({
                "timestamp": segment[-1]["timestamp"],
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "turnover": turnover
            })

        return kline_1m

    def get_quotes(self, symbol: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        sql = "SELECT * FROM quotes WHERE symbol = ?"
        params = [symbol]
        if start_time:
            sql += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            sql += " AND timestamp <= ?"
            params.append(end_time)
        sql += " ORDER BY timestamp ASC"
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(zip([column[0] for column in cursor.description], row)) for row in rows]

    def get_kline_by_period(self, period: str, symbol: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        if period == "1s":
            return self.get_kline_1s(symbol, start_time, end_time)
        elif period == "5s":
            return self.get_kline_5s(symbol, start_time, end_time)
        elif period == "1m":
            return self.get_kline_1m(symbol, start_time, end_time)
        else:
            raise ValueError(f"Unsupported period: {period}")
