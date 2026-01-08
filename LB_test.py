import pandas as pd
import concurrent.futures
import time
import os
from longport.openapi import Config, QuoteContext, Period, AdjustType

# --- CONFIGURATION ---
# Load from environment variables (Recommended)
# export LONGPORT_APP_KEY="your_key"
# export LONGPORT_APP_SECRET="your_secret"
# export LONGPORT_ACCESS_TOKEN="your_token"
config = Config.from_env()
ctx = QuoteContext(config)

OUTPUT_FILE = "stock_liquidity_report.csv"

def fetch_single_stock_history(symbol, static_map):
    """
    Worker function to fetch history for a single stock.
    Run this inside a thread.
    """
    try:
        # 1. Get Static Info
        info = static_map.get(symbol)
        if not info:
            return None

        # 2. Fetch Candlesticks (30 Trading Days)
        candles = ctx.candlesticks(
            symbol=symbol, 
            period=Period.Day, 
            count=30, 
            adjust_type=AdjustType.NoAdjust
        )

        if not candles:
            return None

        # --- CALCULATIONS ---
        last_close = float(candles[-1].close)
        
        # Sums
        total_volume_shares = sum(c.volume for c in candles)
        total_turnover_value = sum(float(c.turnover) for c in candles)
        count = len(candles)

        # Averages
        adv_shares = total_volume_shares / count      # Avg Daily Volume (Shares)
        adt_value = total_turnover_value / count      # Avg Daily Turnover (Value)

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
            "ADT (Value)": adt_value,            # Institutional Liquidity Metric
            "Turnover Rate %": turnover_rate_pct,# Retail Activity Metric
            "Market Cap": market_cap,
            "Exchange": info.exchange,
            "Board": info.board
        }

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def fetch_and_save_data(symbols):
    print(f"Starting fetch for {len(symbols)} stocks...")
    start_time = time.time()
    
    data_list = []

    try:
        # 1. Fetch Static Info in BATCH
        print("Fetching static info batch...")
        static_info_list = ctx.static_info(symbols)
        static_map = {item.symbol: item for item in static_info_list}

        # 2. Multi-threading for Candlesticks
        print(f"Fetching history (Threads=10)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(fetch_single_stock_history, sym, static_map): sym 
                for sym in symbols
            }
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                result = future.result()
                if result:
                    data_list.append(result)

    except Exception as e:
        print(f"Global Error: {e}")

    elapsed = time.time() - start_time
    print(f"Done! Fetched {len(data_list)} stocks in {elapsed:.2f} seconds.")
    
    # --- SAVE TO CSV ---
    if data_list:
        df = pd.DataFrame(data_list)
        
        # Sort by ADT (Liquidity) descending
        df = df.sort_values(by="ADT (Value)", ascending=False)
        
        # Save to CSV
        # encoding='utf-8-sig' ensures Chinese characters (if any) open correctly in Excel
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        print(f"\nSuccess! Data saved to: {os.path.abspath(OUTPUT_FILE)}")
        return df
    else:
        print("No data found.")
        return pd.DataFrame()

if __name__ == "__main__":
    # Your list of stocks
    my_stocks = [
        "700.HK", "9988.HK", "0020.HK", "3690.HK", "1810.HK", 
        "AAPL.US", "NVDA.US", "TSLA.US", "MSFT.US", "AMZN.US",
        "0005.HK", "1299.HK", "0939.HK", "0941.HK", "0388.HK"
    ]
    
    df = fetch_and_save_data(my_stocks)
    
    # Optional: Quick Preview in Console
    if not df.empty:
        pd.options.display.float_format = '{:,.2f}'.format
        print("\n--- Preview (Top 5 by Liquidity) ---")
        print(df[["Symbol", "Name", "ADT (Value)", "Market Cap"]].head(5))