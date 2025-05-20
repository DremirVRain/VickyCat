# 示例：用于测试 data 保存与归档流程
from database import DatabaseManager
from utils.data_cache import DataCache
from utils.archive_manager import ArchiveManager
from datetime import datetime
import asyncio

class DataReplayTester:
    def __init__(self, db_file: str = "market_data_async.db"):
        self.db = DatabaseManager()
        self.cache = DataCache()
        self.archive = ArchiveManager()

    async def replay_quotes(self, symbol: str, start: str, end: str):
        data = self.db.get_quotes(symbol, start, end)
        print(f"[Replay] 回放 {len(data)} 条 quote 数据")

        for row in data:
            timestamp = row["timestamp"]
            print(f"处理: {timestamp} -> {row['price']}")
            self.cache.update_cache(symbol,  row)  # 模拟实时写入
            await asyncio.sleep(0)

    async def run_archive(self):
        self.archive.archive_old_data()

# 示例运行
if __name__ == "__main__":
    tester = DataReplayTester("market_data_async_test.db")
    asyncio.run(tester.replay_quotes("TSLA.US", "2025-05-19 09:30:00", "2025-05-20 21:40:00"))
    asyncio.run(tester.run_archive())
