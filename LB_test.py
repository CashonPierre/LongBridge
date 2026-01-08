from longport.openapi import Config, QuoteContext, Period, AdjustType
import pandas as pd

def fetch_stock_data(symbols):
    # 1. Initialize Context
    # Load credentials from environment variables (Recommended)
    config = Config.from_env() 
    # Or hardcode: config = Config(app_key="...", app_secret="...", access_token="...")
    
    ctx = QuoteContext(config)

    data_list = []

    try:
        # 2. Fetch Static Info (Batch)
        # Docs: https://open.longbridge.com/docs/quote/pull/static
        # This is more efficient than looping for static data
        static_info_list = ctx.static_info(symbols)
        
        # Create a lookup dictionary for static info
        static_map = {item.symbol: item for item in static_info_list}

        # 3. Fetch Candlesticks (Loop)
        # Docs: https://open.longbridge.com/docs/quote/pull/history-candlestick
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            
            # Get basic static details
            info = static_map.get(symbol)
            if not info:
                print(f"Warning: No static info found for {symbol}")
                continue

            # Fetch last 30 daily candlesticks
            # Count=30 gets the last 30 trading days
            candles = ctx.candlesticks(
                symbol=symbol, 
                period=Period.Day, 
                count=30, 
                adjust_type=AdjustType.NoAdjust
            )

            if not candles:
                print(f"Warning: No history found for {symbol}")
                continue

            # --- Data Processing ---
            
            # A. Last Closed Price (Close of the most recent candle)
            last_close_price = candles[-1].close
            
            # B. 30 Days Trading Volume (Sum of volume from last 30 candles)
            # Note: 'volume' in candlesticks is usually share count
            volume_30d = sum(candle.volume for candle in candles)

            # C. Market Cap Calculation
            # Market Cap = Total Shares * Last Price
            # Note: total_shares is often returned as a string or large int
            total_shares = int(info.total_shares) if info.total_shares else 0
            market_cap = total_shares * last_close_price

            # Append to results
            data_list.append({
                "Symbol": symbol,
                "Name": info.name_en, # or info.name_cn
                "Exchange": info.exchange,
                "Board": info.board,
                "Last Close": last_close_price,
                "30D Volume": volume_30d,
                "Total Shares": total_shares,
                "Market Cap": market_cap
            })

    except Exception as e:
        print(f"An error occurred: {e}")
    
    return pd.DataFrame(data_list)

if __name__ == "__main__":
    # Define your list of stocks
    my_stocks = ["700.HK", "AAPL.US", "9988.HK", "NVDA.US"]
    
    df = fetch_stock_data(my_stocks)
    
    # Formatting for better readability
    if not df.empty:
        # Format Market Cap to billions/millions string for display
        pd.options.display.float_format = '{:,.2f}'.format
        print("\n--- Stock Data Report ---")
        print(df)
        
        # Optional: Save to CSV
        # df.to_csv("stock_data.csv", index=False)
    else:
        print("No data fetched.")