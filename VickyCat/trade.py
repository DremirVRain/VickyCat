import datetime
from decimal import Decimal
from typing import List, Optional
from longport.openapi import TradeContext, Config, OrderType, OrderSide, TimeInForceType, OrderStatus, Market

class Trade:
    def __init__(self):
        self.config = Config.from_env()
        self.ctx = TradeContext(self.config)

    def submit_order(self, symbol: str, order_type: OrderType, side: OrderSide, submitted_quantity: Decimal, time_in_force: TimeInForceType, submitted_price: Optional[Decimal] = None, trigger_price: Optional[Decimal] = None, trailing_percent: Optional[Decimal] = None, trailing_amount: Optional[Decimal] = None, limit_offset: Optional[Decimal] = None, expire_date: Optional[datetime.date] = None, outside_rth: Optional[bool] = None, remark: Optional[str] = "") -> dict:
        try:
            return self.ctx.submit_order(
                symbol=symbol,
                order_type=order_type,
                side=side,
                submitted_quantity=submitted_quantity,
                time_in_force=time_in_force,
                submitted_price=submitted_price,
                trigger_price=trigger_price,
                trailing_percent=trailing_percent,
                trailing_amount=trailing_amount,
                limit_offset=limit_offset,
                expire_date=expire_date,
                outside_rth=outside_rth,
                remark=remark,
            )
        except Exception as e:
            return {"error": str(e)}

    def replace_order(self, order_id: str, submitted_quantity: Decimal, price: Optional[Decimal] = None, trigger_price: Optional[Decimal] = None, trailing_amount: Optional[Decimal] = None, trailing_percent: Optional[Decimal] = None, limit_offset: Optional[Decimal] = None, remark: Optional[str] = "") -> dict:
        try:
            return self.ctx.replace_order(
                order_id=order_id,
                quantity=submitted_quantity,
                price=price,
                trigger_price=trigger_price,
                trailing_amount=trailing_amount,
                trailing_percent=trailing_percent,
                limit_offset=limit_offset,
                remark=remark,
            )
        except Exception as e:
            return {"error": str(e)}

    def cancel_order(self, order_id: str) -> dict:
        try:
            return self.ctx.cancel_order(order_id)
        except Exception as e:
            return {"error": str(e)}

    def get_today_orders(self, symbol: Optional[str] = None, status: Optional[List[OrderStatus]] = None, side: Optional[OrderSide] = None, market: Optional[Market] = None) -> dict:
        try:
            return self.ctx.today_orders(
                symbol=symbol,
                status=status,
                side=side,
                market=market,
            )
        except Exception as e:
            return {"error": str(e)}

    def get_history_orders(self, symbol: Optional[str] = None, status: Optional[List[OrderStatus]] = None, side: Optional[OrderSide] = None, market: Optional[Market] = None, start_at: Optional[datetime.datetime] = None, end_at: Optional[datetime.datetime] = None) -> dict:
        try:
            return self.ctx.history_orders(
                symbol=symbol,
                status=status,
                side=side,
                market=market,
                start_at=int(start_at.timestamp()) if start_at else None,
                end_at=int(end_at.timestamp()) if end_at else None,
            )
        except Exception as e:
            return {"error": str(e)}

    def estimate_max_purchase_quantity(self, symbol: str, order_type: OrderType, side: OrderSide, price: Optional[Decimal] = None) -> dict:
        try:
            return self.ctx.estimate_max_purchase_quantity(
                symbol=symbol,
                order_type=order_type,
                side=side,
                price=price,
            )
        except Exception as e:
            return {"error": str(e)}
