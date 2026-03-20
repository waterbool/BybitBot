# Bybit Trading Bot

Python-проект для торговли и исследований по Bybit USDT Perpetual. В репозитории сейчас сосуществуют:

- активный live-стек с Flask UI, live-selector и paper/live execution;
- исследовательский слой с мульти-символьными бэктестами и ранжированием стратегий;
- legacy-модули от более ранней версии односигнального бота.

## Актуальные точки входа

- `main.py --mode live` — базовый CLI-режим live/paper для одного символа.
- `main.py --mode backtest` — быстрый CLI smoke/backfill режим для загрузки данных и проверки сигналов.
- `web_ui.py` — веб-интерфейс и REST/WebSocket API для настройки, бэктеста и управления ботом.
- `scripts/run_strategy_backtests.py` — основной запуск мульти-стратегий и мульти-символьных бэктестов.
- `scripts/run_portfolio_backtests.py` — портфельный отбор кандидатов по score/edge.

## Структура репозитория

- `config.yaml` — основной runtime-конфиг, включая API, риск, ML и live-selector.
- `config/settings.py` — загрузка конфигурации и экспорт runtime settings.
- `bot_controller.py` — orchestration для Web UI и live-selector.
- `web_ui.py` — Flask + Socket.IO сервер.
- `strategy/rules.py` — активные стратегии и фильтры.
- `backtest/` — движок бэктеста, рыночные данные, multi-asset utilities, scoring.
- `live/` — live scanner, trade gate, edge snapshot helpers.
- `ml/` — ML features, model loading, training scripts.
- `data_fetch/` — загрузка свечей, тикеров, funding, open interest.
- `static/` — фронтенд веб-интерфейса.

## Legacy-модули

Эти файлы остались от старой версии проекта и не являются главным active path:

- `strategy.py`
- `indicators.py`
- `risk.py`
- `bybit_client.py`
- `strategy/strategy.py`

## Установка

1. Создай и активируй виртуальное окружение Python 3.10+.
2. Установи зависимости:

```bash
pip install -r requirements.txt
```

## Конфигурация

Основные параметры лежат в `config.yaml`.

- `api.api_key` / `api.api_secret` — ключи Bybit.
- `api.testnet` — `true` для тестовой среды.
- `trading.dry_run` — paper mode без реальных ордеров.
- `live_selector.enabled` — включает multi-symbol selector.
- `live_selector.execution_mode` — `paper` или `live`.

Сейчас секреты хранятся прямо в `config.yaml`, поэтому не коммить реальные ключи.

## Запуск

CLI live/paper:

```bash
python main.py --mode live
```

CLI backtest smoke:

```bash
python main.py --mode backtest
```

Web UI:

```bash
python web_ui.py
```

После запуска Web UI доступен по адресу:

```text
http://localhost:5001
```

## Полезные команды

Мульти-стратегийный бэктест:

```bash
python scripts/run_strategy_backtests.py --help
```

Портфельный бэктест:

```bash
python scripts/run_portfolio_backtests.py --help
```

Обучение ML-модели:

```bash
python ml/train.py
```

## Важные замечания

- `main.py --mode backtest` не заменяет полноценные исследовательские скрипты из `scripts/`.
- В проекте есть смешение active и legacy-кода, поэтому перед изменениями лучше ориентироваться на `bot_controller.py`, `strategy/rules.py`, `backtest/` и `live/`.
- Начинай с `testnet: true` и `dry_run: true`.

## Disclaimer

Торговля фьючерсами несет высокий риск. Сначала проверяй конфигурацию, paper mode и testnet.
