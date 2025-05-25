import datetime
from decimal import Decimal
from typing import Optional, List
import numpy as np

class RiskManager:
    def __init__(self, max_drawdown: float = 0.2, max_position_pct: float = 0.1, confidence_level: float = 0.95):
        """
        初始化风险管理器。

        Args:
            max_drawdown (float): 允许的最大回撤比例，例如 0.2 表示 20%。
            max_position_pct (float): 单笔交易最大仓位占账户余额的比例。
            confidence_level (float): 计算 VaR 和 CVaR 时的置信水平。
        """
        self.max_drawdown = max_drawdown
        self.max_position_pct = max_position_pct
        self.confidence_level = confidence_level
        self.equity_curve: List[float] = []

    def update_equity(self, equity: float):
        """
        更新账户权益曲线。

        Args:
            equity (float): 当前账户总权益。
        """
        self.equity_curve.append(equity)

    def calculate_drawdown(self) -> float:
        """
        计算当前最大回撤。

        Returns:
            float: 当前最大回撤比例。
        """
        if not self.equity_curve:
            return 0.0
        peak = self.equity_curve[0]
        max_drawdown = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def check_drawdown_limit(self) -> bool:
        """
        检查当前回撤是否超过预设的最大回撤限制。

        Returns:
            bool: 如果超过限制，返回 True；否则返回 False。
        """
        current_drawdown = self.calculate_drawdown()
        return current_drawdown > self.max_drawdown

    def calculate_position_size(self, account_balance: float, stop_loss_pct: float) -> float:
        """
        根据账户余额和止损比例计算单笔交易的最大仓位。

        Args:
            account_balance (float): 当前账户余额。
            stop_loss_pct (float): 预设的止损比例，例如 0.02 表示 2%。

        Returns:
            float: 建议的最大仓位金额。
        """
        risk_per_trade = account_balance * self.max_position_pct
        position_size = risk_per_trade / stop_loss_pct
        return position_size

    def calculate_var(self, returns: List[float]) -> float:
        """
        计算给定收益序列的 Value at Risk (VaR)。

        Args:
            returns (List[float]): 收益率序列。

        Returns:
            float: 计算得到的 VaR 值。
        """
        if not returns:
            return 0.0
        var = np.percentile(returns, (1 - self.confidence_level) * 100)
        return var

    def calculate_cvar(self, returns: List[float]) -> float:
        """
        计算给定收益序列的 Conditional Value at Risk (CVaR)。

        Args:
            returns (List[float]): 收益率序列。

        Returns:
            float: 计算得到的 CVaR 值。
        """
        if not returns:
            return 0.0
        var = self.calculate_var(returns)
        cvar = np.mean([r for r in returns if r <= var])
        return cvar
