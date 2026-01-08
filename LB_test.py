import pandas as pd
import concurrent.futures
import time
from longport.openapi import Config, QuoteContext, Period, AdjustType

# Global Context (shared across threads is usually fine for HTTP calls)
# If you encounter issues, move context creation inside the function.
config = Config.from_env() 
ctx = QuoteContext(config)

def fetch_single_stock_history(symbol, static_map):
    """
    Worker function to fetch history for a single stock.
    Run this inside a thread.
    """
    try:
        # 1. Get Static Info from the pre-fetched map
        info = static_map.get(symbol)
        if not info:
            return None

        # 2. Fetch Candlesticks (The slow network part)
        candles = ctx.candlesticks(
            symbol=symbol, 
            period=Period.Day, 
            count=30, 
            adjust_type=AdjustType.NoAdjust
        )

        if not candles:
            return None

        # 3. Calculations
        last_close = float(candles[-1].close)
        
        # Calculate Volume (Shares) and Turnover (Value)
        total_volume_shares = sum(c.volume for c in candles)
        total_turnover_value = sum(float(c.turnover) for c in candles)
        count = len(candles)

        # Averages
        adv_shares = total_volume_shares / count      # Avg Daily Volume
        adt_value = total_turnover_value / count      # Avg Daily Turnover (Liquidity)

        # Turnover Rate %
        total_shares = int(info.total_shares) if info.total_shares else 0
        turnover_rate_pct = 0
        if total_shares > 0:
            turnover_rate_pct = (adv_shares / total_shares) * 100

        # Market Cap
        market_cap = total_shares * last_close

        return {
            "Symbol": symbol,
            "Name": info.name_en,
            "Currency": info.currency,
            "Last Price": last_close,
            "ADT (Value)": adt_value,            # Institutional Liquidity
            "Turnover Rate %": turnover_rate_pct,# Retail/Hype Metric
            "Market Cap": market_cap
        }

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def fetch_market_data_threaded(symbols):
    print(f"Starting fetch for {len(symbols)} stocks...")
    start_time = time.time()
    
    data_list = []

    try:
        # 1. Fetch Static Info in BATCH (Very fast, no need to thread)
        print("Fetching static info batch...")
        static_info_list = ctx.static_info(symbols)
        static_map = {item.symbol: item for item in static_info_list}

        # 2. Multi-threading for Candlesticks
        # max_workers=10 is a safe balance between speed and rate limits
        print("Fetching history in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            # We pass 'static_map' so threads don't need to fetch static info again
            future_to_symbol = {
                executor.submit(fetch_single_stock_history, sym, static_map): sym 
                for sym in symbols
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                result = future.result()
                if result:
                    data_list.append(result)

    except Exception as e:
        print(f"Global Error: {e}")

    elapsed = time.time() - start_time
    print(f"Done! Fetched {len(data_list)} stocks in {elapsed:.2f} seconds.")
    
    return pd.DataFrame(data_list)

if __name__ == "__main__":
    # Test with a longer list to see the speed benefit
    my_stocks = [
        "700.HK", "9988.HK", "0020.HK", "3690.HK", "1810.HK", # HK Tech
        "AAPL.US", "NVDA.US", "TSLA.US", "MSFT.US", "AMZN.US", # US Tech
        "0005.HK", "1299.HK", "0939.HK", "0941.HK", "0388.HK"  # HK Blue chips
    ]
    
    df = fetch_market_data_threaded(my_stocks)
    
    if not df.empty:
        pd.options.display.float_format = '{:,.2f}'.format
        # Sort by Liquidity (ADT)
        df_sorted = df.sort_values(by="ADT (Value)", ascending=False)
        print("\n--- Multi-Threaded Liquidity Report ---")
        print(df_sorted[["Symbol", "Name", "ADT (Value)", "Turnover Rate %", "Market Cap"]])