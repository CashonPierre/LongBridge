import pandas as pd
import concurrent.futures
import time
import os
import csv
import random
from datetime import datetime, timedelta
from longport.openapi import Config, QuoteContext, Period, AdjustType

# --- CONFIGURATION ---
INPUT_FILE = "target_stocks_filtered.xlsx"
OUTPUT_FILE = "full_market_history_3y.csv"
MAX_WORKERS = 4          # Low worker count to prevent Rate Limits (Error 301606)
CANDLE_COUNT = 1000      # Fetch extra candles (approx 4 years) to be safe, then we filter by date

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
    Fetches raw candlestick data with RETRY logic + DATE FILTERING.
    """
    # 1. Symbol Auto-Fix
    if "." not in symbol:
        symbol = f"{symbol}.US"

    # 2. Calculate Cut-off Date (Exactly 3 years ago from today)
    start_date = datetime.now() - timedelta(days=365 * 3)

    # --- RETRY LOGIC ---
    max_retries = 5
    base_delay = 2 
    
    for attempt in range(max_retries):
        try:
            # Request 1000 candles to be safe (covers >3 years even with holidays)
            candles = ctx.candlesticks(symbol, Period.Day, CANDLE_COUNT, AdjustType.NoAdjust)
            
            if not candles:
                return []

            rows = []
            for c in candles:
                # API returns datetime object directly in c.timestamp
                ts = c.timestamp
                
                # --- DATE FILTERING ---
                # Only keep rows where the date is NEWER than start_date
                if ts >= start_date:
                    rows.append({
                        "Symbol": symbol,
                        "Date": ts.strftime('%Y-%m-%d'),
                        "Open": float(c.open),
                        "High": float(c.high),
                        "Low": float(c.low),
                        "Close": float(c.close),
                        "Volume": int(c.volume),
                        "Turnover": float(c.turnover)
                    })
            
            # Success! Add small jitter sleep
            time.sleep(random.uniform(0.1, 0.3))
            return rows

        except Exception as e:
            error_msg = str(e)
            # Handle Rate Limit (301606)
            if "301606" in error_msg or "rate limit" in error_msg.lower():
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"⚠️ Rate limit on {symbol}. Retrying in {wait_time}s... (Attempt {attempt+1})")
                    time.sleep(wait_time)
                    continue 
            
            # Other errors
            print(f"❌ Failed {symbol}: {e}")
            return []
            
    return []

def main():
    # 1. Load Symbols
    print(f"Reading symbols from {INPUT_FILE}...")
    symbols = get_symbols_from_excel(INPUT_FILE)
    
    if not symbols:
        print("No symbols found. Exiting.")
        return

    print(f"Found {len(symbols)} unique symbols. Starting download with {MAX_WORKERS} threads...")

    # 2. Prepare Output CSV
    csv_headers = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume", "Turnover"]
    
    # Logic to create file or append if exists
    file_exists = os.path.exists(OUTPUT_FILE)
    
    try:
        if not file_exists:
            with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
    except Exception as e:
        print(f"Error creating output file: {e}")
        return

    # 3. Multi-threaded Fetching
    start_time = time.time()
    processed_count = 0
    total_rows = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {executor.submit(fetch_stock_history_raw, sym): sym for sym in symbols}
        
        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if data:
                        writer.writerows(data)
                        total_rows += len(data)
                        if processed_count % 10 == 0:
                            f.flush()
                except Exception as exc:
                    print(f"Critical error on {symbol}: {exc}")
                
                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"Progress: {processed_count}/{len(symbols)} stocks processed. ({total_rows} rows saved)")

    elapsed = time.time() - start_time
    print(f"\nDone! Saved {total_rows} rows for {len(symbols)} stocks.")
    print(f"File saved to: {os.path.abspath(OUTPUT_FILE)}")
    print(f"Total time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()