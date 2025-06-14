from base_strategy import BaseStrategy, MarketContext
from strategy_signal import Signal, SignalType
from typing import Optional, List

class MicrostructureStrategy(BaseStrategy):
    """盘口 / 微结构类策略基类"""
    default_params = {
    }

    param_space = {
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)

    def  generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        raise NotImplementedError

class OrderFlowImbalance(MicrostructureStrategy):
    """订单流不平衡策略（买卖方力量）
    条件：当买一挂单量明显大于卖一挂单量（或反之）时产生信号。
    当前订阅等级无盘口数据(bid_volume,ask_volume)，无法使用该策略。
    """
    default_params = {
        "imbalance_ratio_threshold": 2.0
    }

    param_space = {
        "imbalance_ratio_threshold": [1.5, 2.0, 3.0]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.threshold = self.params["imbalance_ratio_threshold"]

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        bid_volume = context.micro_volumes[-1] if len(context.micro_volumes) >= 1 else 0
        ask_volume = context.micro_turnover[-1] if len(context.micro_turnover) >= 1 else 0  # 可根据你定义的 micro 数据字段替换

        if ask_volume == 0 and bid_volume == 0:
            return None

        signal_type = None
        strength = 0.0

        if ask_volume > 0 and bid_volume / ask_volume >= self.threshold:
            signal_type = SignalType.BUY
            strength = min(1.0, (bid_volume / ask_volume - self.threshold + 1) / self.threshold)
        elif bid_volume > 0 and ask_volume / bid_volume >= self.threshold:
            signal_type = SignalType.SELL
            strength = min(1.0, (ask_volume / bid_volume - self.threshold + 1) / self.threshold)

        if signal_type:
            return self.build_signal(
                context.recent_klines[-1],
                signal_type,
                metadata={"bid_volume": bid_volume, "ask_volume": ask_volume, "imbalance_threshold": self.threshold},
                strength=round(strength, 3)
            )

        return None

class PriceSpikeStrategy(MicrostructureStrategy):
    """
    价格急变策略：当价格在短时间内快速上涨或下跌，可能意味着市场异常或情绪波动。

    原理：
    - 当前价格与过去 N 秒前相比涨跌超过设定阈值（如 1%）
    """
    default_params = {
        "threshold": 0.01,
        "window": 5
    }

    param_space = {
        "threshold": [0.005, 0.01, 0.02],
        "window": [3, 5, 10]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.threshold = self.params["threshold"]
        self.window = self.params["window"]
        self.price_history: List[float] = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if not context or len(context.micro_prices) == 0:
            return None

        price = context.micro_prices[-1]
        self.price_history.append(price)
        self.price_history = self.price_history[-self.window:]

        if len(self.price_history) < self.window:
            return None

        old_price = self.price_history[0]
        price_change = (price - old_price) / old_price

        if price_change > self.threshold:
            return self.build_signal(
                context.recent_klines[-1], SignalType.BUY,
                metadata={"price_change": price_change},
                strength=round(price_change, 3)
            )
        elif price_change < -self.threshold:
            return self.build_signal(
                context.recent_klines[-1], SignalType.SELL,
                metadata={"price_change": price_change},
                strength=round(abs(price_change), 3)
            )
        return None


class VolumeSurgeStrategy(MicrostructureStrategy):
    """
    成交量激增策略：当前成交量显著高于短期平均值，可能意味着交易者突然活跃。

    原理：
    - 检查过去 N 条成交量的均值，与当前成交量作比较。
    """
    default_params = {
        "multiplier": 2.0,
        "window": 10
    }

    param_space = {
        "multiplier": [1.5, 2.0, 3.0],
        "window": [5, 10, 15]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.multiplier = self.params["multiplier"]
        self.window = self.params["window"]
        self.vol_history: List[float] = []

    def generate_signal(self, context: Optional[MarketContext] = None) -> Optional[Signal]:
        if not context or len(context.micro_volumes) == 0:
            return None

        volume = context.micro_volumes[-1]
        self.vol_history.append(volume)
        self.vol_history = self.vol_history[-self.window:]

        if len(self.vol_history) < self.window:
            return None

        avg_volume = sum(self.vol_history[:-1]) / (self.window - 1)
        if avg_volume > 0 and volume > avg_volume * self.multiplier:
            strength = volume / avg_volume
            return self.build_signal(
                context.recent_klines[-1],
                SignalType.BUY,
                metadata={"volume": volume, "avg_volume": avg_volume},
                strength=round(strength, 3)
            )
        return None

class TurnoverSpikeStrategy(MicrostructureStrategy):
    """
    成交金额激增策略：成交金额突然放大，可能是大资金入场或主力行为。

    原理：
    - 当前成交额与过去均值对比，大于均值若干倍则视为有效信号。
    """
    default_params = {
        "multiplier": 2.5,
        "window": 10
    }

    param_space = {
        "multiplier": [2.0, 2.5, 3.0],
        "window": [5, 10, 15]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.multiplier = self.params["multiplier"]
        self.turnover_history = []

    def generate_signal(self, context: MarketContext) -> Optional[Signal]:
        turnover = context.micro_turnover[-1] if context.micro_turnover else 0.0
        self.turnover_history.append(turnover)
        self.turnover_history = self.turnover_history[-self.window:]

        if len(self.turnover_history) < self.window:
            return None

        avg_turnover = sum(self.turnover_history[:-1]) / (self.window - 1)
        if avg_turnover > 0 and turnover > avg_turnover * self.multiplier:
            return self.build_signal(
                {"turnover": turnover},
                SignalType.BUY,
                strength=min(turnover / avg_turnover, 2.0),
                metadata={"avg_turnover": avg_turnover}
            )
        return None

class VolumeDryUpStrategy(MicrostructureStrategy):
    """
    成交量收缩策略：成交量突然减少可能预示着趋势即将反转或突破临近。

    原理：
    - 当前成交量低于短期均值的一定比例（如 0.3 倍），视为"量枯"。
    """
    default_params = {
        "ratio": 0.3,
        "window": 10
    }

    param_space = {
        "ratio": [0.2, 0.3, 0.5],
        "window": [5, 10, 15]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.ratio = self.params["ratio"]
        self.volume_history = []

    def generate_signal(self, context: MarketContext) -> Optional[Signal]:
        volume = context.micro_volumes[-1] if context.micro_volumes else 0
        self.volume_history.append(volume)
        self.volume_history = self.volume_history[-self.window:]

        if len(self.volume_history) < self.window:
            return None

        avg_volume = sum(self.volume_history[:-1]) / (self.window - 1)
        if avg_volume > 0 and volume < avg_volume * self.ratio:
            return self.build_signal(
                {"volume": volume},
                SignalType.HOLD,
                strength=1.0 - (volume / avg_volume),
                metadata={"avg_volume": avg_volume}
            )
        return None

class HighFrequencyPingStrategy(MicrostructureStrategy):
    """
    高频成交扰动策略：连续多个极小成交量订单，可能是高频交易测试市场深度。

    原理：
    - 在短时间窗口内出现多笔极小订单（例如 volume < 5），可作为扰动信号。
    """
    default_params = {
        "tiny_volume_threshold": 5,
        "count_threshold": 5,
        "window": 10
    }

    param_space = {
        "tiny_volume_threshold": [1, 3, 5],
        "count_threshold": [3, 5, 7],
        "window": [5, 10]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.tiny_volume_threshold = self.params["tiny_volume_threshold"]
        self.count_threshold = self.params["count_threshold"]
        self.window = self.params["window"]
        self.recent_tiny_trades: List[int] = []

    def generate_signal(self, context: MarketContext) -> Optional[Signal]:
        ts = context.tick_count
        volume = context.micro_volumes[-1] if context.micro_volumes else 0

        if volume <= self.tiny_volume_threshold:
            self.recent_tiny_trades.append(ts)

        self.recent_tiny_trades = [t for t in self.recent_tiny_trades if ts - t <= self.window]

        if len(self.recent_tiny_trades) >= self.count_threshold:
            return self.build_signal(
                {"volume": volume},
                SignalType.HOLD,
                strength=1.0,
                metadata={"tiny_count": len(self.recent_tiny_trades)}
            )
        return None

class SmallTradeClusteringStrategy(MicrostructureStrategy):
    """
    连续微小订单聚集策略：短时间内多次微型订单聚集，或为大单分拆或机器人操作。

    原理：
    - 连续 N 秒内微小订单总数超过阈值。
    - 以成交量小于 threshold 的交易为“微型订单”。
    """
    default_params = {
        "small_volume": 10,
        "window": 3,
        "count_threshold": 4
    }

    param_space = {
        "small_volume": [5, 10, 15],
        "window": [2, 3, 5],
        "count_threshold": [3, 4, 5]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.small_volume = self.params["small_volume"]
        self.window = self.params["window"]
        self.count_threshold = self.params["count_threshold"]
        self.timestamps: List[int] = []

    def generate_signal(self, context: MarketContext) -> Optional[Signal]:
        ts = context.tick_count
        volume = context.micro_volumes[-1] if context.micro_volumes else 0

        if volume <= self.small_volume:
            self.timestamps.append(ts)

        self.timestamps = [t for t in self.timestamps if ts - t <= self.window]

        if len(self.timestamps) >= self.count_threshold:
            return self.build_signal(
                {"volume": volume},
                SignalType.HOLD,
                strength=1.0,
                metadata={"cluster_size": len(self.timestamps)}
            )
        return None

class VolumeSurgeBreakoutStrategy(BaseStrategy):
    """
    成交量或成交金额突然放大，且价格突破最近N根K线的高点或低点，发出信号。
    需要维护历史数据，判断成交量或成交金额是否显著放大（阈值可调）。
    """
    default_params = {
        "window": 20,
        "surge_factor": 2.0
    }

    param_space = {
        "window": [10, 20, 30],
        "surge_factor": [1.5, 2.0, 3.0]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.surge_factor = self.params["surge_factor"]
        self.price_history = []
        self.volume_history = []
        self.turnover_history = []

    def generate_signal(self, context: MarketContext) -> Optional[Signal]:
        price = context.micro_prices[-1] if context.micro_prices else 0
        volume = context.micro_volumes[-1] if context.micro_volumes else 0
        turnover = context.micro_turnover[-1] if context.micro_turnover else 0

        self.price_history.append(price)
        self.volume_history.append(volume)
        self.turnover_history.append(turnover)

        if len(self.price_history) < self.window:
            return None

        highs = self.price_history[-self.window:-1]
        lows = self.price_history[-self.window:-1]

        recent_high = max(highs)
        recent_low = min(lows)

        avg_vol = sum(self.volume_history[-self.window:-1]) / (self.window - 1)
        avg_turn = sum(self.turnover_history[-self.window:-1]) / (self.window - 1)

        vol_surge = volume > avg_vol * self.surge_factor
        turn_surge = turnover > avg_turn * self.surge_factor

        if price > recent_high and (vol_surge or turn_surge):
            return self.build_signal(
                {"price": price},
                SignalType.BUY,
                strength=0.9,
                metadata={"avg_volume": avg_vol, "avg_turnover": avg_turn}
            )
        elif price < recent_low and (vol_surge or turn_surge):
            return self.build_signal(
                {"price": price},
                SignalType.SELL,
                strength=0.9,
                metadata={"avg_volume": avg_vol, "avg_turnover": avg_turn}
            )
        return None


class TurnoverAccelerationStrategy(BaseStrategy):
    """
    成交金额加速度放大策略。
    计算成交金额的一阶导数（差分），再对其做滑动平均平滑，
    当加速度超过阈值，触发信号。
    """
    default_params = {
        "window": 5,
        "smooth_window": 3,
        "threshold": 0.05
    }

    param_space = {
        "window": [3, 5, 7],
        "smooth_window": [2, 3, 4],
        "threshold": [0.03, 0.05, 0.08]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.smooth_window = self.params["smooth_window"]
        self.threshold = self.params["threshold"]
        self.turnovers: List[float] = []

    def generate_signal(self, context: MarketContext) -> Optional[Signal]:
        turnover = context.micro_turnover[-1] if context.micro_turnover else 0.0
        self.turnovers.append(turnover)

        if len(self.turnovers) < self.window + self.smooth_window:
            return None

        diffs = [self.turnovers[i] - self.turnovers[i - 1] for i in range(1, len(self.turnovers))]
        smoothed = [
            sum(diffs[i - self.smooth_window + 1:i + 1]) / self.smooth_window
            for i in range(self.smooth_window - 1, len(diffs))
        ]
        latest_acc = smoothed[-1]

        if abs(latest_acc) < self.threshold:
            return None

        return self.build_signal(
            {"acceleration": latest_acc},
            signal_type=SignalType.BUY if latest_acc > 0 else SignalType.SELL,
            strength=min(abs(latest_acc) / self.threshold, 1.0),
            metadata={"latest_acceleration": latest_acc}
        )


class AggressiveBuyerDetection(BaseStrategy):
    """
    持续小阳线且成交量或成交金额持续上升的买方力量检测策略。
    连续n个周期价格上涨且成交量（或金额）逐步放大。
    """
    default_params = {
        "momentum_window": 5
    }

    param_space = {
        "momentum_window": [3, 5, 7]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["momentum_window"]
        self.prices: List[float] = []
        self.volumes: List[float] = []
        self.turnovers: List[float] = []

    def generate_signal(self, context: MarketContext) -> Optional[Signal]:
        price = context.micro_prices[-1] if context.micro_prices else 0.0
        volume = context.micro_volumes[-1] if context.micro_volumes else 0.0
        turnover = context.micro_turnover[-1] if context.micro_turnover else 0.0

        self.prices.append(price)
        self.volumes.append(volume)
        self.turnovers.append(turnover)

        if len(self.prices) < self.window:
            return None

        p = self.prices[-self.window:]
        v = self.volumes[-self.window:]
        t = self.turnovers[-self.window:]

        price_up = all(p[i] > p[i - 1] for i in range(1, self.window))
        volume_up = all(v[i] >= v[i - 1] for i in range(1, self.window))
        turnover_up = all(t[i] >= t[i - 1] for i in range(1, self.window))

        if price_up and (volume_up or turnover_up):
            return self.build_signal(
                {"price_sequence": p},
                signal_type=SignalType.BUY,
                strength=0.85,
                metadata={"volumes": v, "turnovers": t}
            )
        return None


class ShortTermMomentumStrategy(BaseStrategy):
    """
    连续N秒价格上涨且成交量持续放大的动量跟随策略。
    连续时间段内价格逐步上涨且成交量持续放大，产生买入信号。
    """
    default_params = {
        "window": 5
    }

    param_space = {
        "window": [3, 5, 7]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.window = self.params["window"]
        self.prices: List[float] = []
        self.volumes: List[int] = []

    def generate_signal(self, context: MarketContext) -> Optional[Signal]:
        price = context.micro_prices[-1] if context.micro_prices else 0.0
        volume = context.micro_volumes[-1] if context.micro_volumes else 0

        self.prices.append(price)
        self.volumes.append(volume)

        if len(self.prices) < self.window:
            return None

        p = self.prices[-self.window:]
        v = self.volumes[-self.window:]

        price_up = all(p[i] > p[i - 1] for i in range(1, self.window))
        volume_up = all(v[i] >= v[i - 1] for i in range(1, self.window))

        if price_up and volume_up:
            return self.build_signal(
                {"momentum_prices": p},
                signal_type=SignalType.BUY,
                strength=0.9,
                metadata={"volumes": v}
            )
        return None


class FlatTapeThenSurgeStrategy(BaseStrategy):
    """
    长时间无成交或极低成交，随后突然爆量放大信号。
    监控过去一段时间的成交量是否极低，且当前成交量突然激增。
    """
    default_params = {
        "flat_window": 30,
        "surge_factor": 3.0,
        "low_volume_threshold": 1.0
    }

    param_space = {
        "flat_window": [20, 30, 40],
        "surge_factor": [2.0, 3.0, 4.0],
        "low_volume_threshold": [0.5, 1.0, 2.0]
    }

    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self.flat_window = self.params["flat_window"]
        self.surge_factor = self.params["surge_factor"]
        self.low_volume_threshold = self.params["low_volume_threshold"]
        self.volumes: List[int] = []

    def generate_signal(self, context: MarketContext) -> Optional[Signal]:
        volume = context.micro_volumes[-1] if context.micro_volumes else 0
        self.volumes.append(volume)

        if len(self.volumes) < self.flat_window + 1:
            return None

        recent_volumes = self.volumes[-self.flat_window - 1:-1]
        avg_volume = sum(recent_volumes) / self.flat_window

        if avg_volume < self.low_volume_threshold and volume > avg_volume * self.surge_factor:
            return self.build_signal(
                {"volume": volume},
                signal_type=SignalType.BUY,
                strength=min(volume / (avg_volume * self.surge_factor), 1.0),
                metadata={"avg_volume": avg_volume, "current_volume": volume}
            )
        return None