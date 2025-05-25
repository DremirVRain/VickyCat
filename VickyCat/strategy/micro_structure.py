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
    """订单流不平衡策略（买卖方力量）
    条件：当买一挂单量明显大于卖一挂单量（或反之）时产生信号。
    当前订阅等级无盘口数据(bid_volume,ask_volume)，无法使用该策略。
    """

    def __init__(self, imbalance_ratio_threshold: float = 2.0):
        self.imbalance_ratio_threshold = imbalance_ratio_threshold

    def analyze_microstructure(self, quote: dict) -> Optional[Signal]:
        bid_volume = quote.get("bid_volume", 0)
        ask_volume = quote.get("ask_volume", 0)

        # 防止除以零
        if ask_volume == 0 and bid_volume == 0:
            return None

        signal_type = None
        strength = 0.0

        if ask_volume > 0 and bid_volume / ask_volume >= self.imbalance_ratio_threshold:
            signal_type = SignalType.BUY
            strength = min(1.0, (bid_volume / ask_volume - self.imbalance_ratio_threshold + 1) / self.imbalance_ratio_threshold)

        elif bid_volume > 0 and ask_volume / bid_volume >= self.imbalance_ratio_threshold:
            signal_type = SignalType.SELL
            strength = min(1.0, (ask_volume / bid_volume - self.imbalance_ratio_threshold + 1) / self.imbalance_ratio_threshold)

        if signal_type:
            return Signal(
                symbol=quote["symbol"],
                timestamp=quote["timestamp"],
                signal_type=signal_type,
                strength=round(strength, 3),
                strategy_name=self.__class__.__name__,
                metadata={"bid_volume": bid_volume, "ask_volume": ask_volume}
            )

        return None

class PriceSpikeStrategy(MicrostructureStrategy):
    """
    价格急变策略：当价格在短时间内快速上涨或下跌，可能意味着市场异常或情绪波动。

    原理：
    - 当前价格与过去 N 秒前相比涨跌超过设定阈值（如 1%）
    """
    def __init__(self, threshold: float = 0.01, window: int = 5):
        self.threshold = threshold
        self.window = window
        self.history = []

    def analyze_microstructure(self, quote: dict) -> Optional[Signal]:
        self.history.append((quote["timestamp"], quote["price"]))
        # 保留最近 N 条
        self.history = self.history[-self.window:]

        if len(self.history) < self.window:
            return None
        
        old_price = self.history[0][1]
        price_change = (quote["price"] - old_price) / old_price

        if price_change > self.threshold:
            return Signal(symbol=quote["symbol"], timestamp=quote["timestamp"],
                          signal_type=SignalType.BUY, strength=abs(price_change),
                          strategy_name=self.__class__.__name__, metadata={"change": price_change})
        elif price_change < -self.threshold:
            return Signal(symbol=quote["symbol"], timestamp=quote["timestamp"],
                          signal_type=SignalType.SELL, strength=abs(price_change),
                          strategy_name=self.__class__.__name__, metadata={"change": price_change})
        return None

class VolumeSurgeStrategy(MicrostructureStrategy):
    """
    成交量激增策略：当前成交量显著高于短期平均值，可能意味着交易者突然活跃。

    原理：
    - 检查过去 N 条成交量的均值，与当前成交量作比较。
    """
    def __init__(self, multiplier: float = 2.0, window: int = 10):
        self.multiplier = multiplier
        self.window = window
        self.vol_history = []

    def analyze_microstructure(self, quote: dict) -> Optional[Signal]:
        volume = quote.get("volume", 0)
        self.vol_history.append(volume)
        self.vol_history = self.vol_history[-self.window:]

        if len(self.vol_history) < self.window:
            return None

        avg_volume = sum(self.vol_history[:-1]) / (self.window - 1)
        if avg_volume > 0 and volume > avg_volume * self.multiplier:
            return Signal(symbol=quote["symbol"], timestamp=quote["timestamp"],
                          signal_type=SignalType.BUY, strength=volume / avg_volume,
                          strategy_name=self.__class__.__name__, metadata={"volume": volume, "avg_volume": avg_volume})
        return None

class TurnoverSpikeStrategy(MicrostructureStrategy):
    """
    成交金额激增策略：成交金额突然放大，可能是大资金入场或主力行为。

    原理：
    - 当前成交额与过去均值对比，大于均值若干倍则视为有效信号。
    """
    def __init__(self, multiplier: float = 2.5, window: int = 10):
        self.multiplier = multiplier
        self.window = window
        self.turnover_history = []

    def analyze_microstructure(self, quote: dict) -> Optional[Signal]:
        turnover = quote.get("turnover", 0.0)
        self.turnover_history.append(turnover)
        self.turnover_history = self.turnover_history[-self.window:]

        if len(self.turnover_history) < self.window:
            return None

        avg_turnover = sum(self.turnover_history[:-1]) / (self.window - 1)
        if avg_turnover > 0 and turnover > avg_turnover * self.multiplier:
            return Signal(symbol=quote["symbol"], timestamp=quote["timestamp"],
                          signal_type=SignalType.BUY, strength=turnover / avg_turnover,
                          strategy_name=self.__class__.__name__, metadata={"turnover": turnover, "avg_turnover": avg_turnover})
        return None

class VolumeDryUpStrategy(MicrostructureStrategy):
    """
    成交量收缩策略：成交量突然减少可能预示着趋势即将反转或突破临近。

    原理：
    - 当前成交量低于短期均值的一定比例（如 0.3 倍），视为"量枯"。
    """
    def __init__(self, ratio: float = 0.3, window: int = 10):
        self.ratio = ratio
        self.window = window
        self.volume_history = []

    def analyze_microstructure(self, quote: dict) -> Optional[Signal]:
        volume = quote.get("volume", 0)
        self.volume_history.append(volume)
        self.volume_history = self.volume_history[-self.window:]

        if len(self.volume_history) < self.window:
            return None

        avg_volume = sum(self.volume_history[:-1]) / (self.window - 1)
        if avg_volume > 0 and volume < avg_volume * self.ratio:
            return Signal(symbol=quote["symbol"], timestamp=quote["timestamp"],
                          signal_type=SignalType.HOLD,  # HOLD 表示关注信号
                          strength=1.0 - (volume / avg_volume),
                          strategy_name=self.__class__.__name__, metadata={"volume": volume, "avg_volume": avg_volume})
        return None

class HighFrequencyPingStrategy(MicrostructureStrategy):
    """
    高频成交扰动策略：连续多个极小成交量订单，可能是高频交易测试市场深度。

    原理：
    - 在短时间窗口内出现多笔极小订单（例如 volume < 5），可作为扰动信号。
    """
    def __init__(self, tiny_volume_threshold: int = 5, count_threshold: int = 5, window: int = 10):
        self.tiny_volume_threshold = tiny_volume_threshold
        self.count_threshold = count_threshold
        self.window = window
        self.recent_tiny_trades = []

    def analyze_microstructure(self, quote: dict) -> Optional[Signal]:
        volume = quote.get("volume", 0)
        timestamp = quote["timestamp"]

        if volume <= self.tiny_volume_threshold:
            self.recent_tiny_trades.append(timestamp)

        # 仅保留近 window 秒内的记录
        self.recent_tiny_trades = [ts for ts in self.recent_tiny_trades if timestamp - ts <= self.window]

        if len(self.recent_tiny_trades) >= self.count_threshold:
            return Signal(symbol=quote["symbol"], timestamp=quote["timestamp"],
                          signal_type=SignalType.HOLD,  # 作为干预信号提示
                          strength=1.0,
                          strategy_name=self.__class__.__name__, metadata={"tiny_trade_count": len(self.recent_tiny_trades)})
        return None

class SmallTradeClusteringStrategy(MicrostructureStrategy):
    """
    连续微小订单聚集策略：短时间内多次微型订单聚集，或为大单分拆或机器人操作。

    原理：
    - 连续 N 秒内微小订单总数超过阈值。
    - 以成交量小于 threshold 的交易为“微型订单”。
    """
    def __init__(self, small_volume: int = 10, window: int = 3, count_threshold: int = 4):
        self.small_volume = small_volume
        self.window = window
        self.count_threshold = count_threshold
        self.recent_small_trades = []

    def analyze_microstructure(self, quote: dict) -> Optional[Signal]:
        volume = quote.get("volume", 0)
        timestamp = quote["timestamp"]

        if volume <= self.small_volume:
            self.recent_small_trades.append(timestamp)

        # 仅保留 window 秒内数据
        self.recent_small_trades = [ts for ts in self.recent_small_trades if timestamp - ts <= self.window]

        if len(self.recent_small_trades) >= self.count_threshold:
            return Signal(symbol=quote["symbol"], timestamp=quote["timestamp"],
                          signal_type=SignalType.HOLD,  # 暂无明确方向，建议监控
                          strength=1.0,
                          strategy_name=self.__class__.__name__, metadata={"cluster_size": len(self.recent_small_trades)})
        return None

class VolumeSurgeBreakoutStrategy(BaseStrategy):
    """
    成交量或成交金额突然放大，且价格突破最近N根K线的高点或低点，发出信号。
    需要维护历史数据，判断成交量或成交金额是否显著放大（阈值可调）。
    """
    def __init__(self, symbol: str, window: int = 20, surge_factor: float = 2.0):
        super().__init__(symbol)
        self.window = window
        self.surge_factor = surge_factor
        self.prices: List[float] = []
        self.volumes: List[int] = []
        self.turnovers: List[float] = []

    def generate_signal(self, quote: dict) -> Optional[Signal]:
        price = quote["price"]
        volume = quote["volume"]
        turnover = quote["turnover"]

        self.prices.append(price)
        self.volumes.append(volume)
        self.turnovers.append(turnover)

        if len(self.prices) < self.window:
            return None

        # 取最近window长度的历史数据
        recent_prices = self.prices[-self.window:]
        recent_volumes = self.volumes[-self.window:]
        recent_turnovers = self.turnovers[-self.window:]

        recent_high = max(recent_prices[:-1])  # 排除当前价
        recent_low = min(recent_prices[:-1])

        avg_volume = sum(recent_volumes[:-1]) / (self.window - 1)
        avg_turnover = sum(recent_turnovers[:-1]) / (self.window - 1)

        # 判断是否成交量或金额放大
        volume_surge = volume > avg_volume * self.surge_factor
        turnover_surge = turnover > avg_turnover * self.surge_factor

        metadata = {
            "recent_high": recent_high,
            "recent_low": recent_low,
            "avg_volume": avg_volume,
            "avg_turnover": avg_turnover,
            "volume_surge": volume_surge,
            "turnover_surge": turnover_surge,
        }

        # 价格突破且成交量或金额放大视为买卖信号
        if price > recent_high and (volume_surge or turnover_surge):
            return Signal(
                symbol=quote["symbol"],
                timestamp=quote["timestamp"],
                signal_type=SignalType.BUY,
                strength=0.9,
                strategy_name=self.__class__.__name__,
                metadata=metadata,
            )
        elif price < recent_low and (volume_surge or turnover_surge):
            return Signal(
                symbol=quote["symbol"],
                timestamp=quote["timestamp"],
                signal_type=SignalType.SELL,
                strength=0.9,
                strategy_name=self.__class__.__name__,
                metadata=metadata,
            )
        return None


class TurnoverAccelerationStrategy(BaseStrategy):
    """
    成交金额加速度放大策略。
    计算成交金额的一阶导数（差分），再对其做滑动平均平滑，
    当加速度超过阈值，触发信号。
    """
    def __init__(self, symbol: str, window: int = 5, smooth_window: int = 3, threshold: float = 0.05):
        super().__init__(symbol)
        self.window = window
        self.smooth_window = smooth_window
        self.threshold = threshold
        self.turnovers: List[float] = []

    def generate_signal(self, quote: dict) -> Optional[Signal]:
        turnover = quote["turnover"]
        self.turnovers.append(turnover)

        if len(self.turnovers) < self.window + self.smooth_window:
            return None

        # 计算一阶差分（增量）
        diffs = [self.turnovers[i] - self.turnovers[i - 1] for i in range(1, len(self.turnovers))]

        # 计算滑动平均的加速度
        smoothed_acc = []
        for i in range(self.smooth_window - 1, len(diffs)):
            window_vals = diffs[i - self.smooth_window + 1 : i + 1]
            smoothed_acc.append(sum(window_vals) / self.smooth_window)

        # 最新加速度
        latest_acc = smoothed_acc[-1]

        if latest_acc > self.threshold:
            return Signal(
                symbol=quote["symbol"],
                timestamp=quote["timestamp"],
                signal_type=SignalType.BUY,
                strength=0.8,
                strategy_name=self.__class__.__name__,
                metadata={"latest_acceleration": latest_acc},
            )
        elif latest_acc < -self.threshold:
            return Signal(
                symbol=quote["symbol"],
                timestamp=quote["timestamp"],
                signal_type=SignalType.SELL,
                strength=0.8,
                strategy_name=self.__class__.__name__,
                metadata={"latest_acceleration": latest_acc},
            )
        return None


class AggressiveBuyerDetection(BaseStrategy):
    """
    持续小阳线且成交量或成交金额持续上升的买方力量检测策略。
    连续n个周期价格上涨且成交量（或金额）逐步放大。
    """
    def __init__(self, symbol: str, momentum_window: int = 5):
        super().__init__(symbol)
        self.momentum_window = momentum_window
        self.prices: List[float] = []
        self.volumes: List[int] = []
        self.turnovers: List[float] = []

    def generate_signal(self, quote: dict) -> Optional[Signal]:
        price = quote["price"]
        volume = quote["volume"]
        turnover = quote["turnover"]

        self.prices.append(price)
        self.volumes.append(volume)
        self.turnovers.append(turnover)

        if len(self.prices) < self.momentum_window:
            return None

        recent_prices = self.prices[-self.momentum_window:]
        recent_volumes = self.volumes[-self.momentum_window:]
        recent_turnovers = self.turnovers[-self.momentum_window:]

        # 判断价格是否持续小幅上升（所有涨幅都>0）
        price_increasing = all(
            recent_prices[i] > recent_prices[i - 1] for i in range(1, self.momentum_window)
        )

        # 判断成交量或成交金额是否持续上升
        volume_increasing = all(
            recent_volumes[i] >= recent_volumes[i - 1] for i in range(1, self.momentum_window)
        )
        turnover_increasing = all(
            recent_turnovers[i] >= recent_turnovers[i - 1] for i in range(1, self.momentum_window)
        )

        if price_increasing and (volume_increasing or turnover_increasing):
            return Signal(
                symbol=quote["symbol"],
                timestamp=quote["timestamp"],
                signal_type=SignalType.BUY,
                strength=0.85,
                strategy_name=self.__class__.__name__,
                metadata={
                    "recent_prices": recent_prices,
                    "recent_volumes": recent_volumes,
                    "recent_turnovers": recent_turnovers,
                },
            )
        return None


class ShortTermMomentumStrategy(BaseStrategy):
    """
    连续N秒价格上涨且成交量持续放大的动量跟随策略。
    连续时间段内价格逐步上涨且成交量持续放大，产生买入信号。
    """
    def __init__(self, symbol: str, window: int = 5):
        super().__init__(symbol)
        self.window = window
        self.prices: List[float] = []
        self.volumes: List[int] = []

    def generate_signal(self, quote: dict) -> Optional[Signal]:
        price = quote["price"]
        volume = quote["volume"]

        self.prices.append(price)
        self.volumes.append(volume)

        if len(self.prices) < self.window:
            return None

        recent_prices = self.prices[-self.window:]
        recent_volumes = self.volumes[-self.window:]

        price_up = all(recent_prices[i] > recent_prices[i - 1] for i in range(1, self.window))
        volume_up = all(recent_volumes[i] >= recent_volumes[i - 1] for i in range(1, self.window))

        if price_up and volume_up:
            return Signal(
                symbol=quote["symbol"],
                timestamp=quote["timestamp"],
                signal_type=SignalType.BUY,
                strength=0.9,
                strategy_name=self.__class__.__name__,
                metadata={
                    "recent_prices": recent_prices,
                    "recent_volumes": recent_volumes,
                },
            )
        return None


class FlatTapeThenSurgeStrategy(BaseStrategy):
    """
    长时间无成交或极低成交，随后突然爆量放大信号。
    监控过去一段时间的成交量是否极低，且当前成交量突然激增。
    """
    def __init__(self, symbol: str, flat_window: int = 30, surge_factor: float = 3.0):
        super().__init__(symbol)
        self.flat_window = flat_window
        self.surge_factor = surge_factor
        self.volumes: List[int] = []

    def generate_signal(self, quote: dict) -> Optional[Signal]:
        volume = quote["volume"]
        self.volumes.append(volume)

        if len(self.volumes) < self.flat_window:
            return None

        recent_volumes = self.volumes[-self.flat_window-1:-1]  # 过去flat_window个周期（排除当前）
        avg_volume = sum(recent_volumes) / self.flat_window

        # 判断是否过去成交极低，当前突然爆量
        low_volume = avg_volume < 1  # 你可以调整低成交量阈值
        surge_volume = volume > avg_volume * self.surge_factor

        if low_volume and surge_volume:
            return Signal(
                symbol=quote["symbol"],
                timestamp=quote["timestamp"],
                signal_type=SignalType.BUY,
                strength=0.95,
                strategy_name=self.__class__.__name__,
                metadata={
                    "avg_volume": avg_volume,
                    "current_volume": volume,
                    "surge_factor": self.surge_factor,
                },
            )
        return None