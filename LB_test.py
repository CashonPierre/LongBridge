from longport.openapi import QuoteContext, Config, Period, AdjustType, TradeSessions

config = Config.from_env()
ctx = QuoteContext(config)

resp = ctx.candlesticks(
    "700.HK", Period.Day, 10, AdjustType.NoAdjust, TradeSessions.Intraday)
print(resp)