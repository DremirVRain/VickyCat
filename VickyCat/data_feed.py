import asyncio
from datetime import datetime
from decimal import Decimal
from collections import deque
from longport.openapi import QuoteContext, Config, SubType, PushQuote
from typing import Dict, Any
from database import DatabaseManager
from main import symbols

# 根据 symbols 动态生成 quote_queue，且每个队列最大长度为 1000
quote_queue = {symbol: deque(maxlen=1000) for symbol in symbols}

db_manager = DatabaseManager()

async def save_quote_to_db(quote_data: Dict[str, Any]):
    """保存逐笔行情数据，带 sequence 字段"""
    try:
        await db_manager.save_quote(quote_data)
    except Exception as e:
        print(f"[Error] Saving quote to DB: {e}")


def on_quote(symbol: str, quote: PushQuote):
    """行情推送处理"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    quote_data = {
        "timestamp": timestamp,
        "symbol": symbol,
        "price": float(quote.last_done),
        "volume": int(quote.current_volume) if quote.current_volume else 0,
        "turnover": float(quote.current_turnover) if quote.current_turnover else 0.0
    }
    
    # 判断 symbol 是否在订阅列表中
    if symbol in quote_queue:
        quote_queue[symbol].append(quote_data)
    else:
        print(f"[Warning] Received quote for untracked symbol: {symbol}")


async def start_data_collection():
    """启动行情订阅"""
    try:
        config = Config.from_env()
        ctx = QuoteContext(config)
        ctx.set_on_quote(on_quote)
        ctx.subscribe(symbols, [SubType.Quote], True)
        print("[Info] 开始订阅行情...")
    except Exception as e:
        print(f"[Error] 订阅行情失败: {e}")
        return

    while True:
        try:
            # 批量处理每个 symbol 的数据
            all_data = []
            for symbol, q_queue in quote_queue.items():
                while q_queue:
                    quote_data = q_queue.popleft()
                    all_data.append(quote_data)

            # 批量插入数据库，减少数据库连接频率
            if all_data:
                await db_manager.save_quotes_batch(all_data)

        except Exception as e:
            print(f"[Error] Data collection loop: {e}")

        # 限制循环频率，避免 CPU 占用过高
        await asyncio.sleep(0.1)
