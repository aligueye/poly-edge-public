# poly-edge: Polymarket BTC 5-Min Trading System

## Implementation Plan

---

## What This Is

A Go + Python trading system that trades Polymarket's 5-minute BTC Up/Down binary markets. Two strategies:

1. **Directional ML (primary, 60% capital)** — Uses Binance Futures order flow + technical indicators to predict BTC direction, bets on Polymarket where 0% maker fees make even 53% accuracy profitable.
2. **Market Making (complement, 40% capital)** — Posts two-sided PostOnly quotes around the ML model's fair value to earn maker rebates and capture spread.

### Where the Edge Comes From

The edge is **Polymarket's fee structure**, not the model. A 53-58% accurate 5-min BTC prediction model **loses money** on Binance Futures (fees require 67-85% accuracy to break even). On Polymarket, PostOnly makers pay **0% fees**. Buying YES at $0.52 needs only >52% accuracy to profit. The venue is the edge.

**Bankroll:** $500-$2,000 USDC on Polygon
**Execution:** Cloud VPS, 25-75ms latency
**Target:** $5-15/day on $2K after 3-month learning period

---

## Current Status (as of 2026-02-16)

### What's Built & Working

**Data ingest, core infrastructure, and order execution** are production-ready:

- **WebSocket manager** — generic `Conn` wrapper with auto-reconnect (exp backoff 1s..30s), ping/pong
- **Binance client** — combined stream (depth20, aggTrade, forceOrder), local order book, publishes to event bus
- **Polymarket client** — dual WS: CLOB + RTDS, publishes to event bus
- **Deribit client** — DVOL volatility index via JSON-RPC 2.0, publishes to event bus
- **Event bus** — typed pub/sub with fan-out (7 event types), non-blocking publish
- **TA engine** — RSI(7/14/30), MACD, Stochastic, Bollinger %B, ATR, EMA crossover, momentum, hourly trend from 1-min candles
- **Position manager** — fill recording, weighted avg cost, binary market resolution PnL, strategy capital allocation (60/40)
- **Risk manager** — 4 pre-trade rules (single trade 2%, per-market 10%, total exposure 60%, daily loss 2.5%)
- **Executor** — paper trading mode (simulated fills at limit price, 50-200ms delay) + live mode via CLOB client; batch orders up to 15
- **CLOB HTTP client** — Polymarket CLOB REST API with HMAC-SHA256 L2 auth (place, cancel, cancel-all)
- **Market discovery** — Gamma API polling by slug (`btc-updown-5m-{bucket}`), 60s interval, current + next market tracking
- **Config** — .env loading via godotenv, validation (requires POLY_PRIVATE_KEY, BANKROLL_USDC > 0)
- **main.go** — wires bus, WS clients, TA engine, metrics, position manager, risk, executor, discovery; graceful shutdown
- **Order types** — full order model with validation (Side, OrderType, OrderRequest, OrderResult, FillEvent)
- **Prometheus metrics** — 15 metrics + HTTP server + NewNoop() for test isolation
- **Deploy stack** — Docker Compose (4 services), Dockerfiles, Prometheus config, Grafana dashboard
- **Unit tests** — 22 tests across position (10), risk (5), and executor (7)

### What's Not Built Yet

Strategies, ML pipeline:

| Component | File | Status |
|-----------|------|--------|
| Strategy Router | `internal/strategy/router.go` | Empty stub |
| Directional Strategy | `internal/strategy/directional.go` | Empty stub |
| Market Maker | `internal/strategy/market_maker.go` | Empty stub |
| Research Pipeline | `python/research/*.py` | 1-line docstrings |

---

## Architecture

**Go modular monolith** — hot path: WebSocket connections, TA indicator computation, order construction/signing, position tracking, event routing, Prometheus metrics.

**Python sidecar** — cold path: feature assembly, CatBoost/XGBoost inference, signal generation, backtesting, research.

**IPC:** gRPC over Unix Domain Sockets (<10ms round-trip).

```
+---------------------------------------------------------+
|                 Go Core Binary                           |
|                                                          |
|  +----------+  +----------+  +----------------+         |
|  | WS Mgr   |->| Event Bus|->| Strategy Router|         |
|  | (Poly,   |  |          |  | (directional   |         |
|  |  Binance, |  |          |  |  + MM)         |         |
|  |  Deribit) |  |          |  +-------+--------+         |
|  +----------+  +----------+          |                   |
|                                      |                   |
|  +----------+  +----------+  +-------v--------+         |
|  | Position |<-| Risk     |<-| Order Executor |         |
|  | Manager  |  | Manager  |  +----------------+         |
|  +----------+  +----------+                              |
|                                                          |
|  +--------------------------------------------+         |
|  | TA Engine (RSI, MACD, BB, ATR, Stoch, EMA) |         |
|  +--------------------------------------------+         |
|  +--------------------------------------------+         |
|  | Prometheus /metrics  (port 9090)            |         |
|  +--------------------------------------------+         |
+------------------+--------------------------------------+
                   | gRPC over UDS
         +---------v----------+
         |  Python Sidecar     |
         |  - Feature assembly |
         |  - CatBoost model   |
         |  - Signal generation|
         +--------------------+
```

---

## Third-Party Accounts & Services

### Accounts Needed

| Service | Why | Cost | Notes |
|---------|-----|------|-------|
| **Polymarket** | Trading venue | Free | Polygon wallet + bridge USDC. Generate API key (L2 auth) for CLOB access |
| **Hetzner VPS** (or Vultr) | Execution host | ~$10-15/mo | CPX21 (3 vCPU, 4GB RAM). Ashburn or NYC region |
| **GitHub** | Private repo | Free | |

### Free Data Feeds

| Source | Endpoint | Data |
|--------|----------|------|
| Binance Futures L2 book | `wss://fstream.binance.com/ws/btcusdt@depth20@100ms` | 20-level order book, 100ms |
| Binance Futures trades | `wss://fstream.binance.com/ws/btcusdt@aggTrade` | Trades with buyer/seller flag |
| Binance Futures liquidations | `wss://fstream.binance.com/ws/btcusdt@forceOrder` | Real-time liquidations |
| Binance Futures mark price | `wss://fstream.binance.com/ws/btcusdt@markPrice@1s` | Mark price + funding rate |
| Deribit DVOL | `wss://www.deribit.com/ws/api/v2` | `deribit_volatility_index.btc_usd` |
| Polymarket CLOB WS | `wss://ws-subscriptions-clob.polymarket.com/ws/` | Order book via `market` channel |
| Polymarket RTDS | `wss://ws-live-data.polymarket.com` | Chainlink BTC/USD (settlement oracle) |

### Go Dependencies

```
github.com/polymarket/go-order-utils       # EIP-712 order signing (lowercase, v1.22.6 — for live mode)
# mtt-labs/poly-market-sdk is 404/gone — CLOB client built directly in execution/clob_client.go
github.com/gorilla/websocket               # WebSocket
github.com/cinar/indicator                 # TA indicators (RSI, MACD, BB, ATR, Stoch)
github.com/prometheus/client_golang        # Prometheus
google.golang.org/grpc                     # gRPC
google.golang.org/protobuf                 # Protobuf
github.com/rs/zerolog                      # Structured logging
github.com/joho/godotenv                   # .env config
```

### Python Dependencies

```
py-clob-client catboost xgboost lightgbm numpy pandas
grpcio grpcio-tools scikit-learn optuna shap pyarrow jupyter
```

---

## Repo Structure

```
poly-edge/
├── cmd/poly-edge/main.go                    # DONE — wires bus, WS, TA, metrics, position
├── internal/
│   ├── config/config.go                     # DONE — .env loader, validation
│   ├── ws/
│   │   ├── manager.go                       # DONE — generic WS conn, auto-reconnect
│   │   ├── binance.go                       # DONE — depth20, aggTrade, forceOrder → bus
│   │   ├── polymarket.go                    # DONE — CLOB + RTDS → bus
│   │   └── deribit.go                       # DONE — DVOL → bus
│   ├── eventbus/bus.go                      # DONE — 7 event types, fan-out pub/sub
│   ├── ta/engine.go                         # DONE — RSI, MACD, Stoch, BB, ATR, EMA, momentum
│   ├── strategy/
│   │   ├── router.go                        # STUB
│   │   ├── directional.go                   # STUB
│   │   └── market_maker.go                  # STUB
│   ├── execution/
│   │   ├── executor.go                      # DONE — paper + live modes, RiskChecker interface
│   │   ├── clob_client.go                   # DONE — HMAC-SHA256 L2 auth, place/cancel
│   │   ├── order_types.go                   # DONE — order model + validation
│   │   └── market_discovery.go              # DONE — Gamma API polling by slug
│   ├── position/manager.go                  # DONE — fills, PnL, allocation (60/40)
│   ├── risk/risk.go                         # DONE — 4 pre-trade rules + tests
│   ├── metrics/metrics.go                   # DONE — 15 Prometheus metrics + HTTP server
│   └── grpc/
│       ├── server.go                        # DONE — SignalClient, caches bus data, calls Python
│       └── proto/signals.proto              # DONE — 23-field MarketState + generated stubs
├── python/
│   ├── sidecar/{server,features,model}.py   # DONE — gRPC server, 26 features, CatBoost wrapper
│   ├── research/{collect,label,train,bt}.py # STUBS — 1-line docstrings
│   └── requirements.txt                     # DONE
├── deploy/
│   ├── docker-compose.yml                   # DONE — 4 services
│   ├── Dockerfile.go                        # DONE — multi-stage build
│   ├── Dockerfile.python                    # DONE — python:3.12-slim
│   ├── prometheus.yml                       # DONE — 5s scrape
│   └── grafana/dashboards/trading.json      # DONE
├── .env.example, .gitignore, go.mod, Makefile, README.md  # DONE
```

---

## The Directional ML Strategy

### Signal Pipeline

```
Binance WS --> Go TA Engine --> gRPC MarketState --> Python --> Signal --> Polymarket Order
  L2 book         RSI(7,14,30)      order flow +        CatBoost        PostOnly
  aggTrades       MACD(12/26/9)     TA features         inference       0% fee
  liquidations    Stoch %K(14)
  funding         BB %B, ATR(14)
                  EMA 9v21, 1H trend
```

### Feature Set (25 features)

**Order flow (Python, from raw book/trade data):**
order_flow_imbalance_L5, order_flow_imbalance_L10, normalized_spread, vwap_to_mid_bid, vwap_to_mid_ask, trade_aggressor_ratio_30s/60s/120s, cvd_60s, liquidation_imbalance_60s

**Technical indicators (Go, via cinar/indicator):**
rsi_7, rsi_14, rsi_30, macd_signal, stoch_k, momentum_30s/60s/120s, ema_9_vs_21, atr_14, bollinger_pct_b

**Derived/contextual:**
price_vs_open, time_decay, dvol_level, hourly_trend, mean_reversion_signal

### Trading Logic

- Every 3s: send MarketState to Python, get Signal
- confidence > 0.58 -> PostOnly order in predicted direction (best_ask - $0.01)
- confidence > 0.65 AND time_remaining < 120s -> FOK taker order
- One position per window. Hold to resolution.
- Quarter-Kelly sizing: `size = 0.25 x (edge / (1 - market_price)) x available_capital`

### Key Enhancements

**Multi-timeframe filter (biggest single improvement):** Only take 5-min signals aligned with 1H MACD direction. QuantPedia Nov 2025: Sharpe 0.33 -> 0.80 (+142%), max drawdown halved.

**Mean reversion bias:** BTC has negative autocorrelation at 5-min (~-0.15). After sharp drops, "up" is more likely. The mean_reversion_signal feature captures this.

---

## The Market Making Strategy

**Purpose:** Complement directional with rebate income that's less dependent on accuracy.

- Estimate fair probability from Chainlink price vs window open
- Quote both YES and NO, 4 cents wide, all PostOnly (0% fee)
- Skew to reduce inventory. Refresh on 1.5 cent move or every 30s.
- When directional confidence > 0.65: skew MM quotes in same direction

---

## Key Polymarket API Details

**Market Discovery:** `GET https://gamma-api.polymarket.com/events?slug=btc-updown-5m-{bucket}` (slug-based, NOT tag-based)

**Rate Limits:** 500 orders/sec burst, 60/sec sustained, batch up to 15

**PostOnly:** `orderType: "POSTONLY"` — rejected if would match immediately. Guarantees 0% fee.

**Fee Model:** `taker_fee = shares x 0.0025 x p x (1-p)` | maker_fee = 0 | At 50c: ~1.56% taker, **0% maker**

**Resolution:** Chainlink auto-settlement. Winners get $1.00, losers get $0.00.
