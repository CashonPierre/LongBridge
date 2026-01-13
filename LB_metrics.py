import pandas as pd
import concurrent.futures
import time
import os
from longport.openapi import Config, QuoteContext, Period, AdjustType

# --- CONFIGURATION ---
INPUT_FILE = "target_stocks.xlsx"
OUTPUT_FILE = "stock_liquidity_report.csv"
MAX_WORKERS = 10     # For candlesticks (threading)
BATCH_SIZE = 200     # For Static Info (Safety limit: keep well under 500)

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

def fetch_static_info_in_batches(symbols, batch_size=200):
    """
    Splits the symbol list into smaller chunks to avoid 'Too many symbols' error.
    LongBridge Limit is 500, so we use 200 to be safe.
    """
    full_static_list = []
    total = len(symbols)
    
    print(f"Fetching static info in batches of {batch_size}...")
    
    for i in range(0, total, batch_size):
        chunk = symbols[i : i + batch_size]
        try:
            # Fetch this chunk
            partial_info = ctx.static_info(chunk)
            full_static_list.extend(partial_info)
            # Small sleep to be kind to the rate limiter (optional but good practice)
            time.sleep(0.2)
        except Exception as e:
            print(f"Error fetching batch {i}-{i+batch_size}: {e}")
            
    return full_static_list

def fetch_single_stock_history(symbol, static_map):
    """
    Worker function to fetch candlesticks for a single stock.
    Run this inside a thread.
    """
    try:
        # 1. Get Static Info
        info = static_map.get(symbol)
        if not info:
            return None 

        # 2. Fetch Candlesticks
        # This is the slowest part, so it runs in parallel threads
        candles = ctx.candlesticks(symbol, Period.Day, 30, AdjustType.NoAdjust)
        if not candles:
            return None

        # --- CALCULATIONS ---
        last_close = float(candles[-1].close)
        
        total_volume_shares = sum(c.volume for c in candles)
        total_turnover_value = sum(float(c.turnover) for c in candles)
        count = len(candles)

        # Average Daily Turnover (Value) & Volume (Shares)
        adt_value = total_turnover_value / count
        adv_shares = total_volume_shares / count

        # Market Cap & Turnover Rate
        total_shares = int(info.total_shares) if info.total_shares else 0
        turnover_rate_pct = 0
        if total_shares > 0:
            turnover_rate_pct = (adv_shares / total_shares) * 100

        market_cap = total_shares * last_close

        return {
            "Symbol": symbol,
            "Name": info.name_en,
            "Currency": info.currency,
            "Last Price": last_close,
            "ADT (Value)": adt_value,            
            "Turnover Rate %": turnover_rate_pct,
            "Market Cap": market_cap,
            "Exchange": info.exchange,
            "Board": info.board
        }

    except Exception as e:
        # If rate limit hit, usually returns 429
        # print(f"Error fetching {symbol}: {e}")
        return None

def main():
    # 1. Load Symbols
    print(f"Reading symbols from {INPUT_FILE}...")
    symbols = get_symbols_from_excel(INPUT_FILE)
    
    if not symbols:
        print("No symbols found. Exiting.")
        return

    print(f"Found {len(symbols)} unique symbols.")

    # 2. Fetch Static Info (BATCHED)
    # This fixes the "Too many symbols" error
    try:
        static_info_list = fetch_static_info_in_batches(symbols, BATCH_SIZE)
        static_map = {item.symbol: item for item in static_info_list}
        print(f"Successfully retrieved static info for {len(static_map)} stocks.")
    except Exception as e:
        print(f"Critical Error during static info fetch: {e}")
        return

    # 3. Multi-threaded Fetching (Candlesticks)
    print(f"Fetching candlestick history with {MAX_WORKERS} threads...")
    start_time = time.time()
    data_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks
        future_to_symbol = {
            executor.submit(fetch_single_stock_history, sym, static_map): sym 
            for sym in symbols
        }
        
        # Monitor Progress
        completed = 0
        total_tasks = len(symbols)
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result:
                data_list.append(result)
            
            completed += 1
            if completed % 50 == 0:
                print(f"Progress: {completed}/{total_tasks} stocks processed...")

    elapsed = time.time() - start_time
    print(f"Done! Fetched data for {len(data_list)} stocks in {elapsed:.2f} seconds.")

    # 4. Save to CSV
    if data_list:
        df = pd.DataFrame(data_list)
        
        # Sort by ADT (Liquidity)
        if "ADT (Value)" in df.columns:
            df = df.sort_values(by="ADT (Value)", ascending=False)
        
        # Clean Column Order
        cols = ["Symbol", "Name", "Currency", "Last Price", "ADT (Value)", "Turnover Rate %", "Market Cap", "Exchange", "Board"]
        # Ensure columns exist before selecting
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols]
        
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\nReport saved to: {os.path.abspath(OUTPUT_FILE)}")
    else:
        print("No valid data fetched.")

if __name__ == "__main__":
    main()