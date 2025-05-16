import datetime
from decimal import Decimal
from typing import List, Optional, Callable
import traceback
from longport.openapi import (
    TradeContext, Config, OrderType, OrderSide, TimeInForceType, 
    OrderStatus, Market, TopicType, PushOrderChanged
)

class Trade:
    def __init__(self):
        self.config = Config.from_env()
        self.ctx = TradeContext(self.config)
        self.order_callback: Optional[Callable[[PushOrderChanged], None]] = None

        # 在初始化时就注册推送回调，确保实例化后立即生效
        self.ctx.set_on_order_changed(self.on_order_changed)

    #region 成交

    def get_today_executions(self, symbol: Optional[str] = None, order_id: Optional[str] = None) -> dict:
        try:
            return self.ctx.today_executions(
                symbol=symbol,
                order_id=order_id
            )
        except Exception as e:
            return self.handle_exception(e)

    def get_history_executions(self, symbol: Optional[str] = None, start_at: Optional[datetime.datetime] = None, end_at: Optional[datetime.datetime] = None) -> dict:
        try:
            return self.ctx.history_executions(
                symbol=symbol,
                start_at=int(start_at.timestamp()) if start_at else None,
                end_at=int(end_at.timestamp()) if end_at else None
            )
        except Exception as e:
            return self.handle_exception(e)

    #endregion

    #region 订单

    def estimate_max_purchase_quantity(self, symbol: str, order_type: OrderType, side: OrderSide, price: Optional[Decimal] = None) -> dict:
        try:
            return self.ctx.estimate_max_purchase_quantity(
                symbol=symbol,
                order_type=order_type,
                side=side,
                price=price,
            )
        except Exception as e:
            return self.handle_exception(e)

    def get_today_orders(self, symbol: Optional[str] = None, status: Optional[List[OrderStatus]] = None, side: Optional[OrderSide] = None, market: Optional[Market] = None) -> dict:
        try:
            return self.ctx.today_orders(
                symbol=symbol,
                status=status,
                side=side,
                market=market,
            )
        except Exception as e:
            return self.handle_exception(e)

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
            return self.handle_exception(e)

    def submit_order(
        self, symbol: str, order_type: OrderType, side: OrderSide, quantity: Decimal,
        time_in_force: TimeInForceType, submitted_price: Optional[Decimal] = None,
        trigger_price: Optional[Decimal] = None, trailing_percent: Optional[Decimal] = None,
        trailing_amount: Optional[Decimal] = None, limit_offset: Optional[Decimal] = None,
        expire_date: Optional[datetime.date] = None, remark: Optional[str] = ""
    ) -> dict:
        try:
            return self.ctx.submit_order(
                symbol=symbol,
                order_type=order_type,
                side=side,
                submitted_quantity=quantity,
                time_in_force=time_in_force,
                submitted_price=submitted_price,
                trigger_price=trigger_price,
                trailing_percent=trailing_percent,
                trailing_amount=trailing_amount,
                limit_offset=limit_offset,
                expire_date=expire_date,
                remark=remark,
            )
        except Exception as e:
            return self.handle_exception(e)

    def replace_order(
        self, order_id: str, quantity: Decimal, price: Optional[Decimal] = None,
        trigger_price: Optional[Decimal] = None, trailing_amount: Optional[Decimal] = None,
        trailing_percent: Optional[Decimal] = None, limit_offset: Optional[Decimal] = None,
        remark: Optional[str] = ""
    ) -> dict:
        try:
            return self.ctx.replace_order(
                order_id=order_id,
                submitted_quantity=quantity,
                price=price,
                trigger_price=trigger_price,
                trailing_amount=trailing_amount,
                trailing_percent=trailing_percent,
                limit_offset=limit_offset,
                remark=remark,
            )
        except Exception as e:
            return self.handle_exception(e)

    def cancel_order(self, order_id: str) -> dict:
        try:
            return self.ctx.cancel_order(order_id)
        except Exception as e:
            return self.handle_exception(e)
        
    def order_detail(self, order_id: str) -> dict:
        try:
            return self.ctx.order_detail(order_id)
        except Exception as e:
            return self.handle_exception(e)

    #endregion

    #region 资产

    def get_account_balance(self, currency: Optional[str] = None) -> dict:
        try:
            return self.ctx.account_balance(currency=currency)
        except Exception as e:
            return self.handle_exception(e)

    def get_cash_flow(
        self, 
        start_at: datetime.datetime, 
        end_at: datetime.datetime, 
        business_type: Optional[int] = None,  # 这里需要确认是否一定是整数，是否有具体的业务类型映射表？
        symbol: Optional[str] = None, 
        page: Optional[int] = 1, 
        size: Optional[int] = 50
    ) -> dict:
        try:
            return self.ctx.cash_flow(
                start_at=int(start_at.timestamp()),
                end_at=int(end_at.timestamp()),
                business_type=business_type,
                symbol=symbol,
                page=page,
                size=size
            )
        except Exception as e:
            return self.handle_exception(e)

    def get_stock_positions(self, symbol: Optional[List[str]] = None) -> dict:
        try:
            return self.ctx.stock_positions(symbol=symbol if symbol else [])
        except Exception as e:
            return self.handle_exception(e)

     #endregion

     #region 交易推送

    def subscribe_order_updates(self) -> dict:
        """订阅交易推送。"""
        try:
            response = self.ctx.subscribe([TopicType.Private])
            print(f"Subscribed to order updates: {response}")
            return response or {"status": "No response received"}
        except Exception as e:
            return self.handle_exception(e)

    def unsubscribe_order_updates(self) -> dict:
        """取消订阅交易推送。"""
        try:
            response = self.ctx.unsubscribe([TopicType.Private])
            print(f"Unsubscribed from order updates: {response}")
            return response or {"status": "No response received"}
        except Exception as e:
            return self.handle_exception(e)

    def register_order_callback(self, callback: Callable[[PushOrderChanged], None]) -> None:
        """
        设置订单推送回调处理逻辑。
        Args:
            callback: 回调函数，接收一个 PushOrderChanged 对象作为参数。
        """
        self.order_callback = callback
        self.ctx.set_on_order_changed(self.on_order_changed)
        print("Order callback registered.")

    def on_order_changed(self, event: PushOrderChanged) -> None:
        """
        推送消息回调函数。
        Args:
            event: PushOrderChanged 对象。
        """
        if self.order_callback:
            try:
                self.order_callback(event)
            except Exception as e:
                print(f"Error in order callback: {e}")
                print(traceback.format_exc())
        else:
            print(f"Order update received: {event}")

     #endregion

    def handle_exception(self, e: Exception, debug: bool = True) -> dict:
        """统一异常处理方法，包含详细的异常信息。"""
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc() if debug else "Traceback is hidden in production mode."
        }
        print(f"Exception occurred: {error_info}")
        return error_info
