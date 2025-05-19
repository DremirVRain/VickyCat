import datetime
from datetime import date
from decimal import Decimal
from typing import List, Optional, Callable, Dict
import traceback
from config import ORDER_FEE
from longport.openapi import (
    TradeContext, Config, OrderType, OrderSide, TimeInForceType, 
    OrderStatus, Market, TopicType, PushOrderChanged, Period, AdjustType, TradeSession
)

class Trade:
    def __init__(self):
        self.config = Config.from_env()
        self.ctx = TradeContext(self.config)
        self.order_callback: Optional[Callable[[PushOrderChanged], None]] = None
        self.total_fees = Decimal("0.0")
        
        # 注册推送回调
        self.ctx.set_on_order_changed(self.on_order_changed)

    #region Fee

    def calculate_fee(self, quantity: Decimal) -> Decimal:
        """计算挂单手续费"""
        fee = ORDER_FEE
        self.total_fees += fee
        return fee

    def get_total_fees(self) -> Decimal:
        """获取累计手续费"""
        return self.total_fees

    #endregion

    #region K线数据

    def get_candlesticks(self, symbol: str, period: Period, count: int = 100,
                         adjust_type: AdjustType = AdjustType.NoAdjust, 
                         trade_session: Optional[TradeSession] = TradeSession.All) -> Dict:
        """
        获取指定标的的 K 线数据。

        Args:
            symbol (str): 标的代码，例如 'TSLA.US'
            period (Period): K 线周期，例如 Period.Day
            count (int): 数据数量，最大 1000
            adjust_type (AdjustType): 复权类型，默认为 NoAdjust
            trade_session (Optional[TradeSessions]): 交易时段，默认为 All

        Returns:
            dict: 包含 K 线数据的响应
        """
        try:
            return self.ctx.candlesticks(
                symbol=symbol,
                period=period,
                count=count,
                adjust_type=adjust_type,
                trade_session=trade_session
            )
        except Exception as e:
            return self.handle_exception(e)

    #endregion

    #region 成交

    def get_today_executions(self, symbol: Optional[str] = None, order_id: Optional[str] = None) -> dict:
        """
        获取当天的成交记录。

        Args:
            symbol (Optional[str]): 标的代码（例如 'TSLA.US'），默认返回所有标的
            order_id (Optional[str]): 订单ID，默认返回所有订单的成交记录

        Returns:
            dict: 当天的成交记录数据
        """
        try:
            return self.ctx.today_executions(
                symbol=symbol,
                order_id=order_id
            )
        except Exception as e:
            return self.handle_exception(e)

    def get_history_executions(self, symbol: Optional[str] = None, start_at: Optional[datetime.datetime] = None, end_at: Optional[datetime.datetime] = None) -> dict:
        """
        获取历史的成交记录。

        Args:
            symbol (Optional[str]): 标的代码（例如 'TSLA.US'），默认返回所有标的
            start_at (Optional[datetime.datetime]): 起始时间，默认为 None
            end_at (Optional[datetime.datetime]): 结束时间，默认为 None

        Returns:
            dict: 历史成交记录数据
        """
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
        """
        估算可购买的最大数量。

        Args:
            symbol (str): 标的代码（例如 'TSLA.US'）
            order_type (OrderType): 订单类型（如市价单、限价单等）
            side (OrderSide): 买卖方向（如买入或卖出）
            price (Optional[Decimal]): 如果是限价单，需要指定价格，默认为 None

        Returns:
            dict: 估算的最大购买数量
        """
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
        """
        获取当天的订单。

        Args:
            symbol (Optional[str]): 标的代码（例如 'TSLA.US'），默认返回所有标的
            status (Optional[List[OrderStatus]]): 订单状态（如未成交、已成交等），默认返回所有状态
            side (Optional[OrderSide]): 买卖方向（如买入或卖出），默认返回所有方向
            market (Optional[Market]): 市场类型，默认返回所有市场

        Returns:
            dict: 当天的订单数据
        """
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
        """
        获取历史订单。

        Args:
            symbol (Optional[str]): 标的代码（例如 'TSLA.US'），默认返回所有标的
            status (Optional[List[OrderStatus]]): 订单状态（如未成交、已成交等），默认返回所有状态
            side (Optional[OrderSide]): 买卖方向（如买入或卖出），默认返回所有方向
            market (Optional[Market]): 市场类型，默认返回所有市场
            start_at (Optional[datetime.datetime]): 起始时间，默认为 None
            end_at (Optional[datetime.datetime]): 结束时间，默认为 None

        Returns:
            dict: 历史订单数据
        """
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
        """
        提交订单。

        Args:
            symbol (str): 标的代码，例如 'TSLA.US'
            order_type (OrderType): 订单类型，例如 OrderType.Limit
            side (OrderSide): 买卖方向，例如 OrderSide.Buy
            quantity (Decimal): 下单数量
            time_in_force (TimeInForceType): 订单有效期类型，例如 TimeInForceType.GTC
            submitted_price (Optional[Decimal]): 限价单价格，默认为 None（市价单无需传递）
            trigger_price (Optional[Decimal]): 触发价格，用于止损或跟踪止损单
            trailing_percent (Optional[Decimal]): 跟踪止损百分比，默认为 None
            trailing_amount (Optional[Decimal]): 跟踪止损金额，默认为 None
            limit_offset (Optional[Decimal]): 限价单偏移量（滑点），默认为 None
            expire_date (Optional[datetime.date]): 到期日期，仅适用于 GTC 类型订单
            remark (Optional[str]): 订单备注信息，默认为空字符串

        Returns:
            dict: 订单提交结果
        """
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
        """
        修改订单。

        Args:
            order_id (str): 订单ID
            quantity (Decimal): 修改后的数量
            price (Optional[Decimal]): 修改后的价格，默认为 None（市价单无需传递）
            trigger_price (Optional[Decimal]): 修改后的触发价格，默认为 None
            trailing_amount (Optional[Decimal]): 修改后的跟踪止损金额，默认为 None
            trailing_percent (Optional[Decimal]): 修改后的跟踪止损百分比，默认为 None
            limit_offset (Optional[Decimal]): 修改后的滑点，默认为 None
            remark (Optional[str]): 修改后的备注信息

        Returns:
            dict: 修改订单的结果
        """
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
        """
        取消订单。

        Args:
            order_id (str): 订单ID

        Returns:
            dict: 取消订单的结果
        """
        try:
            return self.ctx.cancel_order(order_id)
        except Exception as e:
            return self.handle_exception(e)
        
    def order_detail(self, order_id: str) -> dict:
        """
        查询订单详情。

        Args:
            order_id (str): 订单ID

        Returns:
            dict: 订单详情数据
        """
        try:
            return self.ctx.order_detail(order_id)
        except Exception as e:
            return self.handle_exception(e)

    #endregion

    #region 行情
    
    def get_trading_days(
        self, 
        market: str, 
        beg_day: str, 
        end_day: str
    ) -> dict:
        """
        获取市场交易日信息

        Args:
            market (str): 市场代码，可选值：US, HK, CN, SG
            beg_day (str): 开始日期，格式 YYMMDD
            end_day (str): 结束日期，格式 YYMMDD

        Returns:
            dict: 包含交易日和半日市信息的字典
        """
        try:
            return self.ctx.trading_days(market, beg_day, end_day)
        except Exception as e:
            return self.handle_exception(e)

    def get_trading_session(self) -> dict:
        """
        获取各市场当日交易时段

        Returns:
            dict: 各市场交易时段数据
        """
        try:
            return self.ctx.trading_session()
        except Exception as e:
            return self.handle_exception(e)

    def is_market_open(
        self, 
        market: str, 
        check_time: Optional[str] = None
    ) -> bool:
        """
        判断指定市场当前是否在交易时段内

        Args:
            market (str): 市场代码（US, HK, CN, SG）
            check_time (Optional[str]): 检查时间，格式：HHMM；若不传，则使用当前时间

        Returns:
            bool: 市场是否处于交易时段
        """
        try:
            # 获取交易时段数据
            session_data = self.get_trading_session()
            market_sessions = next(
                (s["trade_session"] for s in session_data["market_trade_session"] if s["market"] == market), 
                []
            )

            # 获取当前时间
            current_time = check_time if check_time else date.now().strftime("%H%M")
            current_time = int(current_time)

            # 判断是否在交易时段内
            for session in market_sessions:
                if session["beg_time"] <= current_time <= session["end_time"]:
                    return True

            return False

        except Exception as e:
            self.handle_exception(e)
            return False

    #endregion

    #region 资产

    def get_account_balance(self, currency: Optional[str] = None) -> dict:
        """
        获取账户余额。

        Args:
            currency (Optional[str]): 货币类型（如 USD、BTC），默认为 None

        Returns:
            dict: 账户余额数据
        """
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
        """
        获取账户现金流。

        Args:
            start_at (datetime.datetime): 起始时间
            end_at (datetime.datetime): 结束时间
            business_type (Optional[int]): 业务类型，默认为 None
            symbol (Optional[str]): 标的代码，默认为 None
            page (Optional[int]): 当前页码，默认为 1
            size (Optional[int]): 每页的记录数，默认为 50

        Returns:
            dict: 现金流数据
        """
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
        """
        获取账户中的股票持仓。

        Args:
            symbol (Optional[List[str]]): 标的代码列表，默认为 None，返回所有标的的持仓数据

        Returns:
            dict: 股票持仓数据
        """
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
