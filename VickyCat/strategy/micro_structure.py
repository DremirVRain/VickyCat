from base_strategy import BaseStrategy
from strategy_signal import Signal, SignalType
from typing import Optional

class MicrostructureStrategy(BaseStrategy):
    """盘口 / 微结构类策略基类"""
    def analyze_microstructure(self, quote: dict) -> Optional[Signal]:
        raise NotImplementedError

    def generate_signal(self, quote: dict) -> Optional[Signal]:
        return self.analyze_microstructure(quote)

class OrderFlowImbalance(MicrostructureStrategy):
    """订单流不平衡策略（买卖方力量）"""
    def analyze_microstructure(self, quote: dict) -> Optional[Signal]:
        bid_volume = quote.get("bid_volume", 0)
        ask_volume = quote.get("ask_volume", 0)
        if bid_volume > ask_volume * 2:
            return Signal(symbol=quote["symbol"], timestamp=quote["timestamp"],
                          signal_type=SignalType.BUY, strength=0.8,
                          strategy_name=self.__class__.__name__, metadata=quote)
        elif ask_volume > bid_volume * 2:
            return Signal(symbol=quote["symbol"], timestamp=quote["timestamp"],
                          signal_type=SignalType.SELL, strength=0.8,
                          strategy_name=self.__class__.__name__, metadata=quote)
        return None

