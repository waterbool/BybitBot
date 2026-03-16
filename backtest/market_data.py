from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


def symbol_slug(symbol: str) -> str:
    return symbol.strip().lower()


def legacy_symbol_slug(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol.endswith("USDT"):
        return symbol[:-4].lower()
    return symbol.lower()


def _raw_file_candidates(symbol: str, suffix: str) -> list[Path]:
    slug = symbol_slug(symbol)
    legacy = legacy_symbol_slug(symbol)
    candidates = [RAW_DIR / f"{slug}_{suffix}.csv"]
    if legacy != slug:
        candidates.append(RAW_DIR / f"{legacy}_{suffix}.csv")
    return candidates


def resolve_raw_file(symbol: str, suffix: str) -> Path:
    candidates = _raw_file_candidates(symbol, suffix)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_market_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Market data file not found: {path}")
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return df


def load_symbol_frame(symbol: str, suffix: str) -> pd.DataFrame:
    return load_market_frame(resolve_raw_file(symbol, suffix))


def trim_lookback(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if lookback_days <= 0 or df.empty:
        return df.copy()
    cutoff = df["timestamp"].max() - pd.Timedelta(days=lookback_days)
    return df[df["timestamp"] >= cutoff].copy()


def load_base_15m(symbol: str, lookback_days: int) -> pd.DataFrame:
    df = load_symbol_frame(symbol, "15m")
    df = trim_lookback(df, lookback_days)
    return df.set_index("timestamp")


def build_mtf_frame(symbol: str, lookback_days: int, bias_warmup_days: int = 30) -> pd.DataFrame:
    df_15m = load_symbol_frame(symbol, "15m")
    df_1h = load_symbol_frame(symbol, "1h")
    df_15m = trim_lookback(df_15m, lookback_days)
    if not df_15m.empty:
        cutoff = df_15m["timestamp"].min() - pd.Timedelta(days=bias_warmup_days)
        df_1h = df_1h[df_1h["timestamp"] >= cutoff].copy()

    df_1h["ema200_1h"] = df_1h["close"].ewm(span=200, adjust=False).mean()
    df_1h["bias"] = 0
    df_1h.loc[df_1h["close"] > df_1h["ema200_1h"], "bias"] = 1
    df_1h.loc[df_1h["close"] < df_1h["ema200_1h"], "bias"] = -1

    merged = pd.merge_asof(
        df_15m.sort_values("timestamp"),
        df_1h[["timestamp", "bias", "ema200_1h"]].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return merged.set_index("timestamp")


def build_funding_frame(symbol: str, lookback_days: int, context_days: int = 7) -> pd.DataFrame:
    df_15m = load_symbol_frame(symbol, "15m")
    df_funding = load_symbol_frame(symbol, "funding")
    df_oi = load_symbol_frame(symbol, "oi")
    df_15m = trim_lookback(df_15m, lookback_days)
    if not df_15m.empty:
        cutoff = df_15m["timestamp"].min() - pd.Timedelta(days=context_days)
        df_funding = df_funding[df_funding["timestamp"] >= cutoff].copy()
        df_oi = df_oi[df_oi["timestamp"] >= cutoff].copy()

    merged = pd.merge_asof(
        df_15m.sort_values("timestamp"),
        df_funding.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        df_oi.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return merged.set_index("timestamp")
