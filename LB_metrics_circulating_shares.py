import pandas as pd
import time
import os
from longport.openapi import Config, QuoteContext

# --- CONFIGURATION ---
INPUT_FILE = "target_stocks_filtered.xlsx"           # Your excel with symbols in column A
OUTPUT_FILE = "circulating_shares_report.csv"
BATCH_SIZE = 200                            # Safe batch size (API limit ~500)

# Initialize API Context
config = Config.from_env()
ctx = QuoteContext(config)

def get_symbols_from_excel(filepath):
    """Read symbols from first column of Excel, clean & deduplicate"""
    if not os.path.exists(filepath):
        print(f"Error: Input file '{filepath}' not found.")
        return []

    try:
        df = pd.read_excel(filepath, header=None, engine='openpyxl')
        raw_list = df.iloc[:, 0].dropna().astype(str).str.strip()
        
        # Filter out junk like headers or empty
        clean_symbols = [s for s in raw_list if s and s.lower() not in ['nan', 'symbol', 'ticker', '']]
        return list(set(clean_symbols))  # remove duplicates
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return []

def fetch_static_info_in_batches(symbols, batch_size=BATCH_SIZE):
    """Fetch static info in batches to respect rate limits"""
    all_info = []
    total = len(symbols)
    print(f"Fetching circulating shares info for {total} symbols (in batches of {batch_size})...")

    for i in range(0, total, batch_size):
        chunk = symbols[i:i + batch_size]
        try:
            resp = ctx.static_info(chunk)
            all_info.extend(resp)  # resp is list of StaticInfo objects
            print(f"  ✓ Batch {i//batch_size + 1} done ({len(chunk)} symbols)")
            time.sleep(0.3)  # gentle delay to avoid rate limiting
        except Exception as e:
            print(f"Error in batch {i}-{i+len(chunk)}: {e}")
            time.sleep(2)   # longer wait on error

    return all_info

def main():
    # 1. Load symbols
    symbols = get_symbols_from_excel(INPUT_FILE)
    if not symbols:
        print("No valid symbols found. Exiting.")
        return

    print(f"→ Found {len(symbols)} unique symbols")

    # 2. Fetch data
    static_items = fetch_static_info_in_batches(symbols)
    
    if not static_items:
        print("No data retrieved.")
        return

    # 3. Build clean dataframe
    data = []
    for item in static_items:
        try:
            data.append({
                "Symbol": item.symbol,
                "Name_EN": item.name_en,
                "Name_CN": item.name_cn,
                "Exchange": item.exchange,
                "Currency": item.currency,
                "Total Shares": int(item.total_shares) if item.total_shares else 0,
                "Circulating Shares": int(item.circulating_shares) if item.circulating_shares else 0,
                "HK Shares": int(item.hk_shares) if item.hk_shares else 0,  # mostly for HK stocks
                "Lot Size": item.lot_size,
                "Board": item.board
            })
        except AttributeError as e:
            print(f"Skipping malformed item {getattr(item, 'symbol', 'unknown')}: {e}")

    df = pd.DataFrame(data)

    # Optional: Calculate float percentage
    df["Float %"] = (df["Circulating Shares"] / df["Total Shares"] * 100).round(2)
    df["Float %"] = df["Float %"].replace([float('inf'), -float('inf')], 0).fillna(0)

    # Sort by circulating shares descending (biggest float first)
    df = df.sort_values("Circulating Shares", ascending=False)

    # Save
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nDone! Report saved to: {os.path.abspath(OUTPUT_FILE)}")
    print(f"Retrieved info for {len(df)} symbols.")

if __name__ == "__main__":
    main()