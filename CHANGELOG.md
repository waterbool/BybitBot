# Changelog

## Update v0.4.1 — 2026-02-17

### Changed
- Strategy timeframe switched to 30-minute candles (from 15-minute).
- Time-based stop tightened to 4 candles (was 24 in strategy settings and 12 in risk settings).
- Partial take profit adjusted to 1.0 ATR with 50% size (was 1.5 ATR with 30%).
- Trailing stop reduced to 1.5 ATR (was 2.0 ATR).

### Removed
- Market regime filter (ATR% + ADX) and ADX indicator calculation.

### Impact
- Fewer signals due to higher timeframe; each signal reflects longer-term moves.
- Faster exits when trades stall (time stop triggers sooner).
- Earlier partial profit-taking and tighter trailing stop, which can reduce average hold time and drawdown but may cap upside in strong trends.
- No regime gating means signals are no longer blocked by ATR/ADX conditions.
