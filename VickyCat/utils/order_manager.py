from typing import Optional, Callable, Dict, Any, List

class OrderManager:
    def __init__(self, ctx):
        self.ctx = ctx

    async def fetch_data(self, method: Callable, **kwargs) -> Dict[str, Any]:
        """ 通用数据获取方法 """
        try:
            return await method(**kwargs)
        except Exception as e:
            print(f"[Error] 数据获取失败: {e}")
            return {}

    async def get_today_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """ 获取当天订单 """
        return await self.fetch_data(self.ctx.today_orders, symbol=symbol)

    async def get_today_executions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """ 获取当天成交记录 """
        return await self.fetch_data(self.ctx.today_executions, symbol=symbol)

    async def get_history_orders(
        self, symbol: Optional[str] = None, start_at: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """ 获取历史订单 """
        return await self.fetch_data(self.ctx.history_orders, symbol=symbol, start_at=start_at)
