import asyncio
from data_feed import DataFeed
#from strategy_manager import StrategyManager
from config import symbols

async def main():
    data_feed = DataFeed()
    #strategy_manager = StrategyManager(data_feed.db_manager)
    
    # 引入策略处理：绑定策略管理器的接口
    #data_feed.set_quote_callback(strategy_manager.on_quote)

    await data_feed.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("程序终止")