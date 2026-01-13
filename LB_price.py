import pandas as pd
import concurrent.futures
import time
import os
from datetime import datetime, timedelta
from longport.openapi import Config, QuoteContext, Period, AdjustType

# --- CONFIGURATION ---
INPUT_FILE = "target_stocks_filtered.xlsx"
OUTPUT_FILE = "stock_daily_closes.csv"
MAX_WORKERS = 10     # For candlesticks (threading)

# Initialize API Context
config = Config.from_env()
ctx = QuoteContext(config)

def get_symbols_from_excel(filepath):
    """Reads the first column of an Excel file and returns a list of symbols."""
    if not os.path.exists(filepath):
        print(f"Error: Input file '{filepath}' not found.")
        return []

    try:
        # Use openpyxl engine
        df = pd.read_excel(filepath, header=None, engine='openpyxl')
        raw_list = df.iloc[:, 0].tolist()
        
        clean_symbols = []
        for item in raw_list:
            s = str(item).strip()
            # Basic cleanup: remove header rows or empty cells
            if s and s.lower() not in ['nan', 'symbol', 'ticker']:
                clean_symbols.append(s)
        
        # Remove duplicates
        return list(set(clean_symbols))

    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def fetch_single_stock_closes(symbol):
    """
    Worker function to fetch daily closed prices for a single stock over the last 3 years.
    Run this inside a thread.
    """
    try:
        # Fetch Candlesticks (up to 1000, which covers more than 3 years of trading days)
        candles = ctx.candlesticks(symbol, Period.Day, 1000, AdjustType.NoAdjust)
        if not candles:
            return []

        # Calculate start date for 3 years ago
        start_date = datetime.now() - timedelta(days=365 * 3)

        # Collect data
        data = []
        for c in candles:
            ts = c.timestamp
            if ts >= start_date:
                data.append({
                    "Symbol": symbol,
                    "Date": ts.strftime("%Y-%m-%d"),
                    "Close": float(c.close)
                })

        return data

    except Exception as e:
        # print(f"Error fetching {symbol}: {e}")
        return []

def main():
    # 1. Load Symbols
    print(f"Reading symbols from {INPUT_FILE}...")
    symbols = get_symbols_from_excel(INPUT_FILE)
    
    if not symbols:
        print("No symbols found. Exiting.")
        return

    print(f"Found {len(symbols)} unique symbols.")

    # 2. Multi-threaded Fetching (Candlesticks)
    print(f"Fetching daily closes with {MAX_WORKERS} threads...")
    start_time = time.time()
    data_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks
        future_to_symbol = {
            executor.submit(fetch_single_stock_closes, sym): sym 
            for sym in symbols
        }
        
        # Monitor Progress
        completed = 0
        total_tasks = len(symbols)
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result:
                data_list.extend(result)
            
            completed += 1
            if completed % 50 == 0:
                print(f"Progress: {completed}/{total_tasks} stocks processed...")

    elapsed = time.time() - start_time
    print(f"Done! Fetched data for {len(set(d['Symbol'] for d in data_list))} stocks in {elapsed:.2f} seconds.")

    # 3. Save to CSV
    if data_list:
        df = pd.DataFrame(data_list)
        
        # Sort by Symbol and Date
        df = df.sort_values(by=["Symbol", "Date"])
        
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\nReport saved to: {os.path.abspath(OUTPUT_FILE)}")
    else:
        print("No valid data fetched.")

if __name__ == "__main__":
    main()