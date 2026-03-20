import pandas as pd

from strategy.rules import TrendFollowingStrategy


def _build_frame() -> pd.DataFrame:
    dates = pd.date_range(start="2024-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106, 95],
            "high": [101, 102, 103, 104, 105, 106, 107, 95],
            "low": [99, 100, 101, 102, 103, 104, 95, 90],
            "close": [100, 101, 102, 103, 104, 105, 106, 94],
            "SMA_200": [80] * 8,
            "EMA_200": [80] * 8,
            "ATR_20": [2.0] * 8,
            "ATR_14": [2.0] * 8,
            "HighestHigh_7": [101, 102, 103, 104, 105, 106, 107, 107],
            "LowestLow_7": [99, 99, 99, 99, 99, 99, 95, 90],
        },
        index=dates,
    )


def test_trend_following_uses_previous_7_bar_levels():
    print("Testing TrendFollowingStrategy with previous 7-bar levels...")
    strategy = TrendFollowingStrategy(risk_percent=0.01)

    entry_df = _build_frame()
    entry_result = strategy.analyze_market("ETHUSDT", entry_df, 1000.0)
    print(f"Entry result: {entry_result}")
    assert entry_result.action == "BUY"
    assert "LL7 95.00" in entry_result.reason

    strategy.confirm_entry("ETHUSDT", entry_result.entry_price, entry_result.stop_price, entry_result.position_size)

    exit_df = entry_df.copy()
    exit_df.iloc[-1, exit_df.columns.get_loc("open")] = 107
    exit_df.iloc[-1, exit_df.columns.get_loc("high")] = 110
    exit_df.iloc[-1, exit_df.columns.get_loc("low")] = 107
    exit_df.iloc[-1, exit_df.columns.get_loc("close")] = 108
    exit_df.iloc[-1, exit_df.columns.get_loc("HighestHigh_7")] = 110

    exit_result = strategy.analyze_market("ETHUSDT", exit_df, 1000.0)
    print(f"Exit result: {exit_result}")
    assert exit_result.action == "SELL"
    assert "HH7 107.00" in exit_result.reason

    print("SUCCESS: TrendFollowingStrategy uses previous-bar 7-day levels.")


if __name__ == "__main__":
    test_trend_following_uses_previous_7_bar_levels()
