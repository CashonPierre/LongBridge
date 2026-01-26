import pandas as pd
import concurrent.futures
import time
import os
import csv
from datetime import datetime
from longport.openapi import Config, QuoteContext, Period, AdjustType

# --- CONFIGURATION ---
INPUT_FILE = "target_stocks.xlsx"
OUTPUT_FILE = "full_market_history_3y.csv"
MAX_WORKERS = 10        # Number of parallel download threads
CANDLE_COUNT = 800      # ~3.2 years of trading days (252 * 3.2 approx 800)

# Initialize API Context
config = Config.from_env()
ctx = QuoteContext(config)

def get_symbols_from_excel(filepath):
    """Reads the first column of an Excel file and returns a list of symbols."""
    if not os.path.exists(filepath):
        print(f"Error: Input file '{filepath}' not found.")
        return []

    try:
        df = pd.read_excel(filepath, header=None, engine='openpyxl')
        raw_list = df.iloc[:, 0].tolist()
        clean_symbols = [str(s).strip() for s in raw_list if str(s).strip() and str(s).lower() not in ['nan', 'symbol', 'ticker']]
        return list(set(clean_symbols))
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def fetch_stock_history_raw(symbol):
    """
    Fetches raw candlestick data for a single symbol.
    Returns a LIST of dictionaries (one per day).
    """
    try:
        # Fetch last N candles (Daily, No Adjustment)
        # Note: If you need Split/Dividend adjusted data, change to AdjustType.Forward
        candles = ctx.candlesticks(symbol, Period.Day, CANDLE_COUNT, AdjustType.NoAdjust)
        
        if not candles:
            return []

        rows = []
        for c in candles:
            # Convert timestamp to readable date (YYYY-MM-DD)
            # LongPort timestamps are usually unix timestamps
            date_str = c.time.date().strftime('%Y-%m-%d')
            
            rows.append({
                "Symbol": symbol,
                "Date": date_str,
                "Open": float(c.open),
                "High": float(c.high),
                "Low": float(c.low),
                "Close": float(c.close),
                "Volume": int(c.volume),
                "Turnover": float(c.turnover)
            })
        
        return rows

    except Exception as e:
        # print(f"Error fetching {symbol}: {e}") # Optional: Uncomment to debug specific failures
        return []

def main():
    # 1. Load Symbols
    print(f"Reading symbols from {INPUT_FILE}...")
    symbols = get_symbols_from_excel(INPUT_FILE)
    
    if not symbols:
        print("No symbols found. Exiting.")
        return

    print(f"Found {len(symbols)} unique symbols. Starting download...")

    # 2. Prepare Output CSV (Write Header)
    # We write to CSV immediately to save memory
    csv_headers = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume", "Turnover"]
    
    try:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
    except Exception as e:
        print(f"Error creating output file: {e}")
        return

    # 3. Multi-threaded Fetching & Incremental Save
    start_time = time.time()
    processed_count = 0
    total_rows = 0

    # We use a lock to ensure threads don't write to the file at the exact same time
    # However, since we are using 'future.result()' in the main thread, we can write sequentially there.
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_symbol = {executor.submit(fetch_stock_history_raw, sym): sym for sym in symbols}
        
        # Open CSV in append mode for the loop
        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data:
                        writer.writerows(data) # Write this batch of rows
                        total_rows += len(data)
                except Exception as exc:
                    print(f"{symbol} generated an exception: {exc}")
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Progress: {processed_count}/{len(symbols)} stocks processed. ({total_rows} rows saved)")

    elapsed = time.time() - start_time
    print(f"\nDone! Saved {total_rows} rows for {len(symbols)} stocks.")
    print(f"File saved to: {os.path.abspath(OUTPUT_FILE)}")
    print(f"Total time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()