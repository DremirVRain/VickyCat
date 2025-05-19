import asyncio
from data_feed import DataFeed
from strategy import process_strategy
from config import symbols

async def main():
    data_feed = DataFeed()
    await data_feed.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("程序终止")