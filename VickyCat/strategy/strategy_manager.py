from signal import Signal
from typing import List, Union

class StrategyManager:
    def __init__(self, strategies: List):
        self.strategies = strategies

    def generate_signals(self, market_data: Union[dict, list]) -> List[Signal]:
        """遍历所有策略并收集信号"""
        results = []
        for strategy in self.strategies:
            if isinstance(market_data, list):
                for data in market_data:
                    signal = strategy.generate_signal(data)
                    if signal:
                        results.append(signal)
            else:
                signal = strategy.generate_signal(market_data)
                if signal:
                    results.append(signal)
        return results
