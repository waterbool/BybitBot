import requests
import pandas as pd
import time
import logging
import random
from typing import Optional, Union, Dict, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://api.bybit.com"


class NonRetryableHTTPError(Exception):
    pass

def _make_request(endpoint: str, params: Dict, retries: int = 3) -> Dict:
    """
    Helper function to make HTTP requests with retries and rate limiting.
    """
    url = f"{BASE_URL}{endpoint}"
    for i in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 429 or response.status_code >= 500:
                logger.warning(f"Retryable HTTP error {response.status_code}. Retrying in {2 ** i} seconds...")
                time.sleep(2 ** i)
                continue

            if 400 <= response.status_code < 500:
                raise Exception(f"HTTP {response.status_code} for {url}: {response.text}")

            data = response.json()

            if data.get('retCode', 0) != 0:
                logger.error(f"API Error: {data.get('retMsg')}")
                raise Exception(f"Bybit API Error: {data.get('retMsg')}")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}. Retrying ({i+1}/{retries})...")
            time.sleep(1)

    raise Exception(f"Failed to fetch data from {url} after {retries} retries")


def _interval_to_ms(interval: Union[str, int]) -> Optional[int]:
    """
    Convert Bybit interval to milliseconds.
    """
    if isinstance(interval, int):
        return int(interval) * 60 * 1000
    if isinstance(interval, str) and interval.isdigit():
        return int(interval) * 60 * 1000
    interval_map = {
        "D": 24 * 60 * 60 * 1000,
        "W": 7 * 24 * 60 * 60 * 1000,
        "M": 30 * 24 * 60 * 60 * 1000,
    }
    return interval_map.get(interval)


def _request_kline_with_retries(
    endpoint: str,
    params: Dict,
    max_retries: int,
    base_delay_range: tuple = (0.5, 1.0),
    jitter_range: tuple = (0.0, 0.3),
) -> Dict:
    """
    Make a kline request with retry/backoff on rate limits, 5xx, timeouts, or network errors.
    """
    url = f"{BASE_URL}{endpoint}"
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 429 or response.status_code >= 500:
                raise requests.exceptions.HTTPError(f"HTTP {response.status_code}")
            if 400 <= response.status_code < 500:
                raise NonRetryableHTTPError(f"HTTP {response.status_code} for {url}: {response.text}")

            data = response.json()
            if data.get("retCode", 0) != 0:
                raise Exception(f"Bybit API Error: {data.get('retMsg')}")
            return data
        except NonRetryableHTTPError as e:
            raise e
        except (requests.exceptions.RequestException, Exception) as e:
            last_error = e
            if attempt >= max_retries - 1:
                break
            base_delay = random.uniform(*base_delay_range)
            jitter = random.uniform(*jitter_range)
            sleep_s = base_delay * (2 ** attempt) + jitter
            logger.warning(f"Kline request error ({e}). Retrying in {sleep_s:.2f}s ({attempt+1}/{max_retries})...")
            time.sleep(sleep_s)

    raise Exception(
        f"Failed to fetch kline after {max_retries} retries. "
        f"Last error: {last_error}. Params: {params}"
    )


def fetch_klines_paginated(
    symbol: str,
    interval: Union[str, int],
    start_ts_ms: int,
    end_ts_ms: int,
    limit: int = 1000,
    max_retries: int = 5,
    category: str = "linear",
) -> pd.DataFrame:
    """
    Fetch klines with pagination, retries, and safeguards against API issues and infinite loops.
    """
    endpoint = "/v5/market/kline"
    all_rows: List[Dict] = []
    request_count = 0
    empty_retries = 0
    empty_retry_limit = 2
    gap_skip_ms = 6 * 60 * 60 * 1000
    interval_ms = _interval_to_ms(interval)
    last_seen_timestamp: Optional[int] = None
    stagnation_count = 0

    current_start = start_ts_ms
    current_end = end_ts_ms

    logger.info(f"Fetching paginated klines for {symbol} from {start_ts_ms} to {end_ts_ms}")

    while current_start < end_ts_ms and current_end > start_ts_ms:
        if interval_ms:
            window_span = interval_ms * (limit - 1)
            window_start = max(start_ts_ms, current_end - window_span)
            params = {
                "category": category,
                "symbol": symbol,
                "interval": interval,
                "start": window_start,
                "end": current_end,
                "limit": limit,
            }
        else:
            params = {
                "category": category,
                "symbol": symbol,
                "interval": interval,
                "start": current_start,
                "end": end_ts_ms,
                "limit": limit,
            }

        request_count += 1
        logger.info(
            f"[KLINE] Req #{request_count} start={current_start} end={end_ts_ms} limit={limit}"
        )

        response_data = _request_kline_with_retries(endpoint, params, max_retries=max_retries)
        result_list = response_data.get("result", {}).get("list", [])

        if not result_list:
            if empty_retries < empty_retry_limit:
                empty_retries += 1
                logger.warning(
                    f"[KLINE] Empty response (retry {empty_retries}/{empty_retry_limit}) "
                    f"for {symbol} {interval} start={current_start} end={current_end}"
                )
                time.sleep(0.5)
                continue
            if interval_ms:
                logger.warning(
                    f"[KLINE] No data returned for range ending at {current_end}. "
                    f"Skipping backward by {gap_skip_ms // 3600000}h."
                )
                current_end = max(start_ts_ms, current_end - gap_skip_ms)
            else:
                logger.warning(
                    f"[KLINE] No data returned for range starting at {current_start}. "
                    f"Skipping ahead by {gap_skip_ms // 3600000}h."
                )
                current_start += gap_skip_ms
            empty_retries = 0
            continue

        empty_retries = 0

        batch_df = pd.DataFrame(
            result_list,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )
        if batch_df.empty:
            current_start += gap_skip_ms
            continue

        batch_df["timestamp"] = batch_df["timestamp"].astype(int)
        batch_df = batch_df.sort_values("timestamp")

        rows = batch_df.to_dict("records")
        new_rows = []
        for row in rows:
            ts = int(row["timestamp"])
            if start_ts_ms <= ts <= end_ts_ms:
                if interval_ms:
                    if ts <= current_end:
                        new_rows.append(row)
                else:
                    if ts >= current_start:
                        new_rows.append(row)

        if not new_rows:
            logger.warning(
                f"[KLINE] No new rows after filtering for start={current_start} end={current_end}. "
                "Advancing window to avoid stall."
            )
            if interval_ms:
                current_end = max(start_ts_ms, current_end - interval_ms)
            else:
                current_start += 1
            continue

        all_rows.extend(new_rows)
        new_last_ts = int(new_rows[-1]["timestamp"])
        new_first_ts = int(new_rows[0]["timestamp"])

        if interval_ms:
            if last_seen_timestamp is not None and new_first_ts >= last_seen_timestamp:
                stagnation_count += 1
                logger.warning(
                    f"[KLINE] Pagination stalled: new_first_ts={new_first_ts} "
                    f">= last_seen_first={last_seen_timestamp} (count={stagnation_count})"
                )
                current_end = max(start_ts_ms, last_seen_timestamp - interval_ms)
                if stagnation_count >= 3:
                    raise Exception(
                        f"Pagination stalled for {symbol} {interval}. "
                        f"start={start_ts_ms} end={end_ts_ms}"
                    )
                continue
            stagnation_count = 0
            last_seen_timestamp = new_first_ts
        else:
            if last_seen_timestamp is not None and new_last_ts <= last_seen_timestamp:
                stagnation_count += 1
                logger.warning(
                    f"[KLINE] Pagination stalled: new_last_ts={new_last_ts} "
                    f"<= last_seen={last_seen_timestamp} (count={stagnation_count})"
                )
                current_start = last_seen_timestamp + 1
                if stagnation_count >= 3:
                    raise Exception(
                        f"Pagination stalled for {symbol} {interval}. "
                        f"start={start_ts_ms} end={end_ts_ms}"
                    )
                continue
            stagnation_count = 0
            last_seen_timestamp = new_last_ts

        logger.info(
            f"[KLINE] Received {len(new_rows)} candles; total={len(all_rows)}; "
            f"first_ts={new_first_ts} last_ts={new_last_ts}"
        )

        if interval_ms:
            if new_first_ts <= start_ts_ms:
                break
            if len(result_list) < limit and new_first_ts <= start_ts_ms:
                break
            current_end = new_first_ts - 1
        else:
            if new_last_ts >= end_ts_ms:
                break
            if len(result_list) < limit:
                break
            current_start = new_last_ts + 1
        time.sleep(0.1)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "datetime"])

    df = pd.DataFrame(all_rows)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    cols = ["open", "high", "low", "close", "volume"]
    df[cols] = df[cols].astype(float)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    df = df.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp").reset_index(drop=True)

    if interval_ms:
        diffs = df["timestamp"].diff().dropna()
        gaps = diffs[diffs != interval_ms]
        if not gaps.empty:
            logger.warning(f"[KLINE] Detected {len(gaps)} time gaps (expected {interval_ms} ms).")
            sample = gaps.head(20)
            for idx, diff in sample.items():
                ts_before = int(df.loc[idx - 1, "timestamp"])
                ts_after = int(df.loc[idx, "timestamp"])
                diff_minutes = diff / 60000.0
                logger.warning(
                    f"[KLINE] Gap: {ts_before} -> {ts_after} ({diff_minutes:.2f} min)"
                )
    else:
        logger.warning(f"[KLINE] Interval '{interval}' not numeric; skipping gap check.")

    requested_days = (end_ts_ms - start_ts_ms) / 86400000.0
    received_days = (len(df) * interval_ms / 86400000.0) if interval_ms else 0.0
    start_dt = df["datetime"].iloc[0] if not df.empty else None
    end_dt = df["datetime"].iloc[-1] if not df.empty else None
    logger.info(
        f"[KLINE] requested_days={requested_days:.2f} "
        f"received_candles={len(df)} "
        f"received_days_estimate={received_days:.2f} "
        f"start_datetime={start_dt} end_datetime={end_dt}"
    )
    if requested_days - received_days > 0.1:
        logger.warning(
            "[KLINE] Получено меньше данных, чем запрошено. Возможно лимит API или проблема пагинации. "
            "Рекомендуется включить pagination (fetch_klines_paginated)."
        )

    if interval_ms and len(df) >= 2:
        expected_count = int(((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) // interval_ms) + 1)
        missing = max(expected_count - len(df), 0)
        missing_ratio = missing / expected_count if expected_count > 0 else 0.0
        if missing_ratio > 0:
            logger.warning(f"[KLINE] Missing candles ratio: {missing_ratio:.4f} ({missing}/{expected_count})")
            diffs = df["timestamp"].diff()
            gap_indices = diffs[diffs > interval_ms].index[:20]
            for idx in gap_indices:
                ts_before = int(df.loc[idx - 1, "timestamp"])
                ts_after = int(df.loc[idx, "timestamp"])
                diff_minutes = (ts_after - ts_before) / 60000.0
                logger.warning(
                    f"[KLINE] Missing gap: {ts_before} -> {ts_after} ({diff_minutes:.2f} min)"
                )

    return df

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
    return fetch_klines_paginated(
        symbol=symbol,
        interval=interval,
        start_ts_ms=start_time,
        end_ts_ms=end_time,
        limit=1000,
        max_retries=5,
        category=category,
    )

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
