import asyncio
from data_feed import start_data_collection
from data_feed import symbols
from data_processor import start_data_processing
from strategy import process_strategy

async def main():
    # 启动数据采集、数据处理、策略执行
    await asyncio.gather(
        start_data_collection(),
        start_data_processing(),
        process_strategy(symbols[0]),
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("程序终止")
