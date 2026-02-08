import requests
import pandas as pd
import time
import logging
from typing import Optional, Union, Dict, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://api.bybit.com"

def _make_request(endpoint: str, params: Dict, retries: int = 3) -> Dict:
    """
    Helper function to make HTTP requests with retries and rate limiting.
    """
    url = f"{BASE_URL}{endpoint}"
    for i in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            
            # Simple rate limit handling: if headers indicate low limit remaining, sleep
            # Note: Bybit v5 returns X-Bapi-Limit-Status etc.
            # For simplicity, we just catch 429 or check status.
            
            if response.status_code == 429:
                logger.warning(f"Rate limit hit. Retrying in {2 ** i} seconds...")
                time.sleep(2 ** i)
                continue
                
            data = response.json()
            
            if data['retCode'] != 0:
                logger.error(f"API Error: {data['retMsg']}")
                raise Exception(f"Bybit API Error: {data['retMsg']}")
                
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}. Retrying ({i+1}/{retries})...")
            time.sleep(1)
            
    raise Exception(f"Failed to fetch data from {url} after {retries} retries")

def fetch_historical_klines(symbol: str, interval: str, start_time: int, end_time: int, category: str = 'linear') -> pd.DataFrame:
    """
    Fetch historical klines (candles) for a symbol within a time range.
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '1', '5', '60', 'D')
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
        category: Product category ('linear', 'spot', 'inverse'). Defaults to 'linear'.
    
    Returns:
        pandas.DataFrame: DataFrame with columns [timestamp, open, high, low, close, volume]
    """
    endpoint = "/v5/market/kline"
    limit = 1000 # Max limit for Bybit v5 kline
    all_data = []
    
    # Bybit returns data in descending order (latest first) by default usually,
    # but let's check docs logic. v5 market/kline: 'data sorted in ascending order if start/end used?' 
    # Actually v5 usually returns in reverse order for some endpoints, but let's strictly control via loop.
    # We will fetch chunk by chunk.
    # To handle pagination safely for wide ranges:
    # We can iterate from start_time forward or end_time backward.
    # Let's iterate from start_time forward.
    
    current_start = start_time
    
    logger.info(f"Fetching historical klines for {symbol} from {start_time} to {end_time}")

    while current_start < end_time:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "start": current_start,
            "end": end_time,
            "limit": limit
        }
        
        response_data = _make_request(endpoint, params)
        result_list = response_data.get('result', {}).get('list', [])
        
        if not result_list:
            break
            
        # Bybit v5 kline returns: [startTime, open, high, low, close, volume, turnover]
        # The list is usually sorted by time descending (newest first). 
        # But if we provide start/end, we get what falls in that range.
        # We need to ensure we collect them all.
        
        # Let's handle the sorting. If it comes descending:
        # [timestamp_N, ..., timestamp_M] where N > M.
        # If we asked for start=100, end=500. It might give us 500...100.
        
        # Parse current batch
        batch_df = pd.DataFrame(result_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        
        # Convert timestamp to int for sorting
        batch_df['timestamp'] = batch_df['timestamp'].astype(int)
        
        # Sort ascending to find the last timestamp covering our specific requested range
        batch_df = batch_df.sort_values('timestamp')
        
        rows = batch_df.to_dict('records')
        
        if not rows:
            break

        # Filter out anything we already have (to prevent overlap issues if any) and verify range
        new_rows = []
        for row in rows:
            ts = int(row['timestamp'])
            if ts >= current_start and ts <= end_time:
               new_rows.append(row)
        
        if not new_rows:
            # If we got data but nothing new that meets criteria (shouldn't happen with correct logic), break to avoid infinite loop
            break
            
        all_data.extend(new_rows)
        
        last_ts = int(new_rows[-1]['timestamp'])
        
        # Calculate next start time. 
        # interval is string, e.g. '1' (minute). We need to convert to ms to add to last_ts.
        # However, relying on adding interval can be tricky with 'D' or 'W'.
        # Safer way: update current_start to last_ts + 1ms (or just > last_ts).
        
        if last_ts >= end_time:
            break
            
        # Advance current_start
        # If the API returned fewer items than limit, we are likely done.
        if len(result_list) < limit:
            break
            
        # Update current_start to be strictly greater than the last received timestamp
        current_start = last_ts + 1
        
        # Respect rate limits (though _make_request handles 429, we can be nice)
        time.sleep(0.1) 
        
    if all_data:
        df = pd.DataFrame(all_data)
        # Select and reorder columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Convert numeric columns
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].astype(float)
        
        # Convert timestamp to datetime (optional per requirements, but good for display)
        # Requirement: "timestamp (datetime or int)". Let's keep int as raw and add a datetime column or just return int?
        # Requirement says "returning ... timestamp (datetime or int)". 
        # Let's return datetime object as it's more Pandas-friendly for timeseries.
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Clean up duplications if any
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        return df
    else:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

def get_latest_kline(symbol: str, interval: str, category: str = 'linear') -> Dict:
    """
    Get the latest (most recent) kline.
    
    Args:
        symbol: Trading symbol
        interval: Kline interval
        category: Product category
        
    Returns:
        dict: Dictionary with kline data (timestamp, open, high, low, close, volume)
    """
    endpoint = "/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": 1
    }
    
    data = _make_request(endpoint, params)
    result_list = data.get('result', {}).get('list', [])
    
    if result_list:
        kline = result_list[0]
        # kline is [startTime, open, high, low, close, volume, ...]
        return {
            "timestamp": int(kline[0]),
            "open": float(kline[1]),
            "high": float(kline[2]),
            "low": float(kline[3]),
            "close": float(kline[4]),
            "volume": float(kline[5]),
            "datetime": datetime.fromtimestamp(int(kline[0])/1000)
        }
    else:
        return {}

if __name__ == "__main__":
    # Example usage
    symbol = "BTCUSDT"
    interval = "1" # 1 minute
    
    # Fetch last 2 hours of data
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - (2 * 60 * 60 * 1000) # 2 hours ago
    
    print(f"--- Fetching historical klines for {symbol} ---")
    df = fetch_historical_klines(symbol, interval, start_ts, end_ts)
    print(f"Downloaded {len(df)} candles.")
    if not df.empty:
        print("Head:")
        print(df.head())
        print("\nTail:")
        print(df.tail())
        
    print(f"\n--- Fetching latest kline for {symbol} ---")
    latest = get_latest_kline(symbol, interval)
    print(latest)
