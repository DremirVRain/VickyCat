from datetime import datetime

MOCK_NOW = None

def now():
    return MOCK_NOW or datetime.now()
