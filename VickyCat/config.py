from decimal import Decimal

symbols = ["TSLA.US", "TSDD.US"]

# 交易手续费
ORDER_FEE = Decimal("1.05")  # 每次挂单手续费

# 持仓周期控制
MIN_HOLDING_PERIOD = 60  # 最小持仓周期，单位：秒

# 交易时间段配置
TRADE_SESSION = {
    "US": {
        "PRE_TRADE": (400, 930),
        "NORMAL_TRADE": (930, 1600),
        "POST_TRADE": (1600, 2000)
    },
    "HK": {
        "NORMAL_TRADE": (930, 1200),
        "AFTERNOON_TRADE": (1300, 1600)
    },
    "CN": {
        "NORMAL_TRADE": (930, 1130),
        "AFTERNOON_TRADE": (1300, 1457)
    }
}