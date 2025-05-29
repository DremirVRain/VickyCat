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
        更新 1s 数据缓存，处理跳空，更新现有蜡烛或插入新数据。

        Args:
            symbol (str): 股票代码。
            data (Dict[str, Any]): 逐笔行情数据，包含 'timestamp', 'price', 'volume', 'turnover'。
        """
        cache = self.cache[symbol]
        curr_time = datetime.datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")

        # 如果缓存为空，直接插入第一根蜡烛
        if not cache:
            cache.append(self._create_new_candle(curr_time, data))
            return

        last_candle = cache[-1]
        last_time = datetime.datetime.strptime(last_candle["timestamp"], "%Y-%m-%d %H:%M:%S")
        delta_seconds = int((curr_time - last_time).total_seconds())

        if delta_seconds == 0:
            # 同一秒内更新
            last_candle["close"] = data["price"]
            last_candle["high"] = max(last_candle["high"], data["price"])
            last_candle["low"] = min(last_candle["low"], data["price"])
            last_candle["volume"] += data["volume"]
            last_candle["turnover"] += data["turnover"]
        elif delta_seconds == 1:
            # 正常递增1秒，添加新蜡烛
            cache.append(self._create_new_candle(curr_time, data))
        elif delta_seconds > 1:
            # 有跳空，补齐中间空窗蜡烛
            for i in range(1, delta_seconds):
                missing_time = last_time + datetime.timedelta(seconds=i)
                fake_candle = {
                    "timestamp": missing_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": last_candle["close"],
                    "high": last_candle["close"],
                    "low": last_candle["close"],
                    "close": last_candle["close"],
                    "volume": 0,
                    "turnover": 0
                }
                cache.append(fake_candle)
            # 插入当前有效蜡烛
            cache.append(self._create_new_candle(curr_time, data))

    def _create_new_candle(self, dt: datetime.datetime, data: Dict[str, Any]) -> Dict[str, Any]:
        """构建标准 1s 蜡烛"""
        return {
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "open": data["price"],
            "high": data["price"],
            "low": data["price"],
            "close": data["price"],
            "volume": data["volume"],
            "turnover": data["turnover"]
        }

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
