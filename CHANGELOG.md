# Changelog

## Update v0.6.2 — 2026-03-21

### English
- Added a robust configuration search runner that evaluates randomized presets across multiple symbols and rolling validation windows.
- Added saved preset storage for the three selected candidate variants and a dedicated rerun script for repeatable comparisons.
- Improved `signal_score` so rescoring stays consistent when ATR, RSI, and Bollinger periods are tuned during variant search and replay.

### Русский
- Добавлен раннер robust-поиска конфигураций, который проверяет случайные пресеты по нескольким символам и rolling-окнам валидации.
- Добавлено сохранение трёх выбранных candidate-вариантов и отдельный скрипт для их повторного прогона и воспроизводимого сравнения.
- Улучшен `signal_score`: теперь пересчёт корректно учитывает тюнингованные периоды ATR, RSI и Bollinger при поиске и повторном прогоне вариантов.

## Update v0.6.1 — 2026-03-20

### English
- Fixed the `TrendFollowingStrategy` 7-bar breakout logic to use previous completed candles, so pullback entries and impulse exits are no longer impossible.
- Unified the single-symbol baseline pipeline across CLI, Web UI, and bot controller through a shared helper module for closed-candle preparation, indicators, ML filtering, and signal generation.
- Removed the live scanner side effect that temporarily mutated global `ML_ENABLED`, reducing cross-request drift risk.
- Updated repository docs and added a regression test for the corrected 7-bar level behavior.

### Русский
- Исправлена логика 7-свечных уровней в `TrendFollowingStrategy`: теперь используются предыдущие закрытые свечи, поэтому входы на pullback и impulse-exit больше не являются недостижимыми.
- Выровнен single-symbol baseline pipeline между CLI, Web UI и контроллером бота через общий helper для закрытых свечей, индикаторов, ML-фильтра и генерации сигналов.
- Убран побочный эффект live-scanner, который временно менял глобальный `ML_ENABLED`, что снижает риск дрейфа между параллельными запусками и запросами.
- Обновлена документация репозитория и добавлен регрессионный тест для исправленного поведения 7-свечных уровней.

## Update v0.6.0 — 2026-03-17

### English
- Added a frozen `edge snapshot` pipeline for live selection, including snapshot build/load helpers and a CLI builder script.
- Added a multi-symbol closed-candle live scanner with per-strategy data preparation for baseline, MTF, and funding-based setups.
- Added a live `trade gate` with snapshot freshness checks, signal staleness checks, spread validation, and daily entry caps.
- Extended the bot controller with selector-driven `paper/live` execution, symbol-aware position management, and richer trade metadata.
- Enabled the new live selector in `paper` mode by default so the system can observe real-time candidates without sending real orders.

### Русский
- Добавлен пайплайн `edge snapshot` для live-отбора: сбор, загрузка и CLI-скрипт для построения исторического снимка качества.
- Добавлен мульти-символьный live-сканер по закрытым свечам с отдельной подготовкой данных для baseline, MTF и funding-стратегий.
- Добавлен live `trade gate` с проверками свежести snapshot, устаревания сигнала, спреда и дневных лимитов на новые входы.
- Контроллер бота расширен для selector-driven `paper/live` исполнения, управления позициями по конкретным символам и сохранения расширенной метаинформации по сделкам.
- Новый live selector включён в режиме `paper` по умолчанию, чтобы наблюдать реальные сигналы без отправки реальных ордеров.

## Update v0.5.1 — 2026-03-17

### English
- Added historical `edge_score` for each `symbol + strategy` pair based on profit factor, win rate, drawdown, sample size, and net PnL.
- Upgraded portfolio selection from raw `signal_score` ranking to composite `selection_score = signal_score + edge_score` with configurable weights.
- Added stricter quality-first filters: minimum edge threshold, per-symbol daily cap, and per-strategy daily cap.
- Tightened default portfolio selection to fewer trades per day and higher minimum score requirements.

### Русский
- Добавлен исторический `edge_score` для каждой пары `symbol + strategy` на основе profit factor, win rate, drawdown, размера выборки и net PnL.
- Портфельный отбор переведён с простого ранжирования по `signal_score` на составной `selection_score = signal_score + edge_score` с настраиваемыми весами.
- Добавлены более жёсткие quality-first фильтры: минимальный порог edge, дневной лимит на символ и дневной лимит на стратегию.
- Значения по умолчанию для портфельного отбора ужесточены в сторону меньшего числа сделок и более высокого минимального качества сигналов.

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
