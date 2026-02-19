# poly-edge

Trading system for Polymarket's 5-minute BTC Up/Down binary markets.

## Quick Start

```bash
cp .env.example .env
# Edit .env with your Polymarket credentials
make build
make run
# Ctrl+C for graceful shutdown
```

## Architecture

```
                    ┌──────────────┐
                    │   main.go    │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Binance  │ │Polymarket│ │ Deribit  │
        │ WS Feed  │ │ WS Feeds │ │ WS Feed  │
        └──────────┘ └──────────┘ └──────────┘
         depth20      CLOB book    DVOL index
         aggTrade     RTDS price
         forceOrder

              │            │            │
              └────────────┼────────────┘
                           ▼
                    ┌──────────────┐
                    │  Event Bus   │
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐      ┌───────────┐
   │ TA Engine│     │  Market  │      │Directional│
   │ RSI/MACD │     │  Maker   │      │  Strategy │
   └──────────┘     └──────────┘      └───────────┘
                           │
                    ┌──────┴───────┐
                    │  Execution   │  EIP-712 signing
                    │  CLOB Client │  SOCKS5 proxy
                    │  + Discovery │  Gamma API
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Position │ │   Risk   │ │ Metrics  │
        └──────────┘ └──────────┘ └──────────┘

        ┌─────────────────────────────────────┐
        │  Python Sidecar (gRPC over UDS)     │
        │  CatBoost ML model for signals      │
        └─────────────────────────────────────┘
```

## Project Structure

```
cmd/poly-edge/main.go          Entry point, wires all components, SIGINT/SIGTERM shutdown
internal/
  config/config.go              .env loader, validation
  ws/
    manager.go                  Generic WS conn wrapper, auto-reconnect, ping/pong
    binance.go                  Binance Futures: depth20, aggTrade, forceOrder → bus
    polymarket.go               CLOB order book + RTDS Chainlink BTC/USD → bus
    deribit.go                  DVOL volatility index via JSON-RPC → bus
  eventbus/bus.go               Typed event bus, 7 events, fan-out pub/sub
  ta/engine.go                  RSI, MACD, Stoch, BB, ATR, EMA, momentum, trend
  strategy/
    router.go                   Concurrent strategy dispatcher (60/40 allocation)
    directional.go              ML signal-driven bets, quarter-Kelly, PostOnly/FOK
    market_maker.go             Two-sided quoting, inventory skew, 4c spread
  execution/
    order_types.go              Order model + validation
    executor.go                 Paper + live order execution, batch orders
    eip712.go                   EIP-712 order signing, tick-size-aware amount rounding
    clob_client.go              Polymarket CLOB REST, HMAC auth, SOCKS5 proxy, tick size cache
    market_discovery.go         Gamma API polling, current + next market
  position/manager.go           Fill tracking, resolution PnL, allocation (60/40)
  risk/risk.go                  4 pre-trade risk rules
  metrics/metrics.go            15 Prometheus metrics + HTTP server
  grpc/
    server.go                   SignalClient: caches bus data, calls Python over UDS
    proto/signals.proto         23-field MarketState + AggTrade + Liquidation + Signal
python/
  sidecar/
    features.py                 26 features (OFI, spread, VWAP, aggressor, TA, derived)
    model.py                    XGBoost/CatBoost/LightGBM loader via training_meta.json
    server.py                   gRPC server on UDS
  research/
    backfill.py                 30-day historical data collection (vectorized TA)
    collect_training_data.py    Live data collector (all 26 features incl. order flow)
    train_model.py              CatBoost/XGBoost/LightGBM with Optuna, SHAP
    backtest.py                 Threshold sweep, quarter-Kelly, PnL curves
deploy/                         Docker Compose, Prometheus, Alertmanager, Grafana
scripts/                        Utility scripts (balance, cancel, cashout)
```

## Live Data Feeds

| Source | Stream | Update Rate | Data |
|--------|--------|-------------|------|
| Binance Futures | `btcusdt@depth20@100ms` | ~100ms | 20-level order book |
| Binance Futures | `btcusdt@aggTrade` | continuous | Aggregated trades |
| Binance Futures | `btcusdt@forceOrder` | on event | Liquidations |
| Polymarket CLOB | market channel | on event | Order book snapshots |
| Polymarket RTDS | `crypto_prices_chainlink` | ~1s | BTC/USD Chainlink price |
| Deribit | `deribit_volatility_index.btc_usd` | ~1s | DVOL implied volatility |

All connections auto-reconnect with exponential backoff (1s, 2s, 4s, ... max 30s).

## Configuration

Copy `.env.example` to `.env` and fill in your credentials. Required fields:

- `POLY_PRIVATE_KEY` — Polymarket wallet private key
- `BANKROLL_USDC` — Total USDC bankroll (must be > 0)

See `.env.example` for all options with descriptions.

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make build` | Compile Go binary |
| `make run` | Run via `go run` |
| `make live` | Run with timestamped log file |
| `make test` | Run Go tests |
| `make proto` | Generate gRPC stubs (Go + Python) |
| `make backfill` | Collect 30 days of historical training data |
| `make train` | Train ML models (CatBoost/XGBoost/LightGBM) |
| `make backtest` | Run backtesting framework |
| `make collect` | Start live data collector (systemd service) |
| `make collect stop` | Stop collector |
| `make collect restart` | Restart collector |
| `make collect status` | Show collector PID, uptime, data size |
| `make collect logs` | Tail collector logs |
| `make balance` | Check wallet balance |
| `make cancel` | Cancel all open orders |
| `make cashout` | Cash out to external wallet |
| `make docker-up` | Start all services |
| `make docker-down` | Stop all services |

## Dependencies

**Go:** gorilla/websocket, godotenv, zerolog, cinar/indicator/v2, prometheus/client_golang

**Python:** py-clob-client, catboost, xgboost, lightgbm, optuna, numpy, pandas, grpcio, scikit-learn, shap, matplotlib

## Sprint Progress

- [x] Sprint 0 — Repo scaffolding, config, WebSocket connections, auto-reconnect, graceful shutdown
- [x] Sprint 1 — Event bus, TA engine, position manager, risk manager, metrics
- [x] Sprint 2 — Order execution, CLOB client, market discovery
- [x] Sprint 3 — gRPC bridge, Python feature engineering, 26 features
- [x] Sprint 4 — Training data collection pipeline
- [x] Sprint 5 — Model training (CatBoost/Optuna), backtesting framework
- [x] Sprint 6 — Directional strategy, market making, strategy router
- [x] Sprint 7 — Deployment hardening (Docker, Prometheus alerts, Grafana dashboards, Telegram)
- [x] Sprint 8 — EIP-712 signing, CLOB client live execution, SOCKS5 proxy, tick-size rounding
- [x] Sprint 9 — Fill tracking, market resolution, async MM, startup reconciliation
- [ ] Sprint 10 — Retrain models with live data (7+ days of collector data)
