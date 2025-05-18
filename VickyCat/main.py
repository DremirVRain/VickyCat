import asyncio
from data_feed import start_data_collection
from strategy import process_strategy

# 定义交易标的
symbols = ["TSLA.US", "TSDD.US"]

async def main():
    # 启动数据采集和策略执行
    await asyncio.gather(
        start_data_collection(),
        *[process_strategy(symbol) for symbol in symbols]
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("程序终止")