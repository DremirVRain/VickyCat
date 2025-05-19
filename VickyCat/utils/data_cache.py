from collections import defaultdict, deque
from typing import Dict, List, Any
import datetime

class DataCache:
    def __init__(self, max_length: int = 1000):
        """
        初始化缓存，仅保存 1s 数据，用于实时生成 5s 和 1m 数据。
        """
        self.cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_length))

    def update_cache(self, symbol: str, data: Dict[str, Any]):
        """
        更新 1s 数据缓存，更新现有蜡烛数据或插入新数据。

        Args:
            symbol (str): 股票代码。
            data (Dict[str, Any]): 逐笔行情数据，包含 'timestamp', 'price', 'volume', 'turnover'。
        """
        cache = self.cache[symbol]
        timestamp = data["timestamp"]

        # 如果缓存为空，直接插入数据
        if not cache:
            new_candle = {
                "timestamp": timestamp,
                "open": data["price"],
                "high": data["price"],
                "low": data["price"],
                "close": data["price"],
                "volume": data["volume"],
                "turnover": data["turnover"]
            }
            cache.append(new_candle)
            return

        # 获取当前未闭合的蜡烛
        last_candle = cache[-1]

        # 如果时间戳相同，则更新现有蜡烛数据
        if last_candle["timestamp"] == timestamp:
            last_candle["close"] = data["price"]
            last_candle["high"] = max(last_candle["high"], data["price"])
            last_candle["low"] = min(last_candle["low"], data["price"])
            last_candle["volume"] += data["volume"]
            last_candle["turnover"] += data["turnover"]
        else:
            # 插入新的蜡烛数据
            new_candle = {
                "timestamp": timestamp,
                "open": data["price"],
                "high": data["price"],
                "low": data["price"],
                "close": data["price"],
                "volume": data["volume"],
                "turnover": data["turnover"]
            }
            cache.append(new_candle)

    def get_cached_data(self, symbol: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """
        获取 1s 数据缓存，用于生成 5s 和 1m 数据。

        Args:
            symbol (str): 股票代码。
            start_time (str): 起始时间戳，格式 "%Y-%m-%d %H:%M:%S"。
            end_time (str): 结束时间戳，格式 "%Y-%m-%d %H:%M:%S"。

        Returns:
            List[Dict[str, Any]]: 满足条件的数据列表。
        """
        start_ts = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_ts = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

        return [
            item for item in self.cache[symbol]
            if start_ts <= datetime.datetime.strptime(item["timestamp"], "%Y-%m-%d %H:%M:%S") <= end_ts
        ]