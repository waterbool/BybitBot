# Changelog

## Update v0.5.0 — 2026-03-17

### English
- Added multi-symbol raw data export with per-symbol files for instruments like `ETHUSDT`, `BTCUSDT`, and `SOLUSDT`.
- Added signal quality scoring and backtest metadata, including `signal_score`, symbol, strategy name, and score components for each position.
- Added a multi-symbol strategy backtest runner that builds consolidated JSON and CSV reports across several instruments.
- Added portfolio-level selection of the top scored trades per day to support a quality-over-quantity workflow.
- Refactored market-data loading and multi-asset backtest utilities into reusable modules.

### Русский
- Добавлен экспорт рыночных данных по нескольким инструментам с отдельными файлами для каждого символа, например `ETHUSDT`, `BTCUSDT` и `SOLUSDT`.
- Добавлен скоринг качества сигналов и расширенная метаинформация в бэктесте: `signal_score`, символ, название стратегии и компоненты оценки для каждой позиции.
- Добавлен мульти-символьный раннер бэктестов, который формирует общие JSON- и CSV-отчёты сразу по нескольким инструментам.
- Добавлен портфельный отбор лучших сделок по score за каждый день для режима, где качество сигналов важнее количества входов.
- Логика загрузки рыночных данных и мульти-ассетного бэктеста вынесена в переиспользуемые модули.

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
