from decimal import Decimal

symbols = ["TSLA.US", "TSDD.US"]

# ����������
ORDER_FEE = Decimal("1.05")  # ÿ�ιҵ�������

# �ֲ����ڿ���
MIN_HOLDING_PERIOD = 60  # ��С�ֲ����ڣ���λ����

# ����ʱ�������
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