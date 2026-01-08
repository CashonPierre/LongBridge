import pandas as pd
import concurrent.futures
import time
import os
from longport.openapi import Config, QuoteContext, Period, AdjustType

# --- CONFIGURATION ---
INPUT_FILE = "raw_data_symbol_list.xlsx"       # Name of your input Excel file
OUTPUT_FILE = "stock_liquidity_report.csv" # Name of the output CSV
MAX_WORKERS = 10                        # Thread count (10 is safe for API limits)

# Initialize API Context
config = Config.from_env()
ctx = QuoteContext(config)

def get_symbols_from_excel(filepath):
    """
    Reads the first column of an Excel file and returns a list of symbols.
    """
    if not os.path.exists(filepath):
        print(f"Error: Input file '{filepath}' not found.")
        return []

    try:
        # Read Excel (engine='openpyxl' is standard for .xlsx)
        # header=None reads the first row as data, header=0 reads it as a header.
        # We use header=None to grab everything, then clean it up.
        df = pd.read_excel(filepath, header=None, engine='openpyxl')
        
        # Select the first column (index 0)
        raw_list = df.iloc[:, 0].tolist()
        
        # CLEANING:
        # 1. Convert to string
        # 2. Strip whitespace
        # 3. Filter out NaN (empty cells) or common headers like "Symbol"
        clean_symbols = []
        for item in raw_list:
            s = str(item).strip()
            # Skip empty cells, 'nan', or a header row that says "Symbol" or "Ticker"
            if s and s.lower() != 'nan' and s.lower() != 'symbol' and s.lower() != 'ticker':
                clean_symbols.append(s)
        
        # Remove duplicates just in case
        return list(set(clean_symbols))

    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def fetch_single_stock_history(symbol, static_map):
    """Worker function to fetch data for one stock."""
    try:
        # 1. Get Static Info
        info = static_map.get(symbol)
        if not info:
            return None # Skip if symbol invalid

        # 2. Fetch Candlesticks
        candles = ctx.candlesticks(symbol, Period.Day, 30, AdjustType.NoAdjust)
        if not candles:
            return None # Skip if no history

        # --- CALCULATIONS ---
        last_close = float(candles[-1].close)
        
        total_volume_shares = sum(c.volume for c in candles)
        total_turnover_value = sum(float(c.turnover) for c in candles)
        count = len(candles)

        adv_shares = total_volume_shares / count
        adt_value = total_turnover_value / count

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
        print(f"Error fetching {symbol}: {e}")
        return None

def main():
    # 1. Load Symbols from Excel
    print(f"Reading symbols from {INPUT_FILE}...")
    symbols = get_symbols_from_excel(INPUT_FILE)
    
    if not symbols:
        print("No symbols found or file is empty. Exiting.")
        return

    print(f"Found {len(symbols)} unique symbols. (e.g., {symbols[:3]}...)")

    # 2. Fetch Static Info (Batch)
    print("Fetching static info...")
    try:
        static_info_list = ctx.static_info(symbols)
        static_map = {item.symbol: item for item in static_info_list}
    except Exception as e:
        print(f"Failed to fetch static info. Check your API Key. Error: {e}")
        return

    # 3. Multi-threaded Fetching
    print(f"Fetching history with {MAX_WORKERS} threads...")
    start_time = time.time()
    data_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {
            executor.submit(fetch_single_stock_history, sym, static_map): sym 
            for sym in symbols
        }
        
        # Optional: Simple counter for progress
        completed = 0
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result:
                data_list.append(result)
            completed += 1
            if completed % 10 == 0:
                print(f"Progress: {completed}/{len(symbols)}...")

    elapsed = time.time() - start_time
    print(f"Done! Fetched {len(data_list)} stocks in {elapsed:.2f} seconds.")

    # 4. Save to CSV
    if data_list:
        df = pd.DataFrame(data_list)
        df = df.sort_values(by="ADT (Value)", ascending=False)
        
        # Reorder columns for readability
        cols = ["Symbol", "Name", "Currency", "Last Price", "ADT (Value)", "Turnover Rate %", "Market Cap", "Exchange", "Board"]
        df = df[cols]
        
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\nReport saved successfully to: {os.path.abspath(OUTPUT_FILE)}")
    else:
        print("No valid data fetched.")

if __name__ == "__main__":
    main()