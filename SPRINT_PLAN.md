# poly-edge: Sprint Plan & Claude Code Prompts

## How to Use This Document

Each sprint contains a **Prompt** to copy-paste into Claude Code and a **Done When** checklist. Sprints are sequential. Each assumes prior sprints are complete.

**Strategy recap:** Two strategies only.
1. **Directional ML (primary, 60% capital)** — Predict BTC 5-min direction using Binance order flow + TA indicators, bet on Polymarket where 0% maker fees make 53% accuracy profitable.
2. **Market Making (complement, 40% capital)** — Two-sided PostOnly quoting for rebates + spread capture.

---

## Current State (as of 2026-02-16)

Sprints 0–7 are **complete**. Full trading pipeline is operational in paper mode with monitoring:

| Component | Status | Notes |
|-----------|--------|-------|
| WS manager (auto-reconnect, ping/pong) | DONE | `internal/ws/manager.go` (221 LOC) |
| Binance client (depth20, aggTrade, forceOrder) | DONE | `internal/ws/binance.go` (~200 LOC), local order book, publishes to event bus |
| Polymarket dual WS (CLOB + RTDS) | DONE | `internal/ws/polymarket.go` (~260 LOC), publishes to event bus |
| Deribit DVOL | DONE | `internal/ws/deribit.go` (~130 LOC), publishes to event bus |
| Config loading + validation | DONE | `internal/config/config.go` (112 LOC) |
| main.go with full wiring | DONE | `cmd/poly-edge/main.go` — bus, WS, TA, metrics, position, risk, executor, discovery, gRPC |
| Order type definitions + validation | DONE | `internal/execution/order_types.go` (111 LOC) |
| Event bus (typed pub/sub) | DONE | `internal/eventbus/bus.go` — 7 event types, fan-out, non-blocking |
| TA engine | DONE | `internal/ta/engine.go` — RSI, MACD, Stochastic, BB, ATR, EMA, momentum, trend |
| Position manager | DONE | `internal/position/manager.go` — fills, resolution PnL, allocation (60/40) |
| Risk manager | DONE | `internal/risk/risk.go` — 4 pre-trade rules + unit tests |
| Executor (paper + live) | DONE | `internal/execution/executor.go` — paper fills, live via CLOB client, batch orders |
| CLOB HTTP client | DONE | `internal/execution/clob_client.go` — HMAC-SHA256 L2 auth, place/cancel orders |
| Market discovery | DONE | `internal/execution/market_discovery.go` — Gamma API polling by slug, 60s interval |
| Prometheus metrics | DONE | `internal/metrics/metrics.go` — 15 metrics + HTTP server + NewNoop() for tests |
| Deploy stack (Docker, Prometheus, Grafana) | DONE | `deploy/` — Compose, Dockerfiles, prometheus.yml, grafana dashboard |
| Proto (full MarketState) | DONE | `internal/grpc/proto/signals.proto` — 23-field MarketState + AggTrade + Liquidation + Signal |
| Go gRPC SignalClient | DONE | `internal/grpc/server.go` — caches bus data, assembles MarketState, calls Python over UDS |
| Python features.py | DONE | `python/sidecar/features.py` — 26 features (10 OFI, 12 TA, 4 derived) |
| Python model.py | DONE | `python/sidecar/model.py` — CatBoost wrapper, hold if no model.cbm |
| Python server.py | DONE | `python/sidecar/server.py` — gRPC server on UDS |
| Makefile | DONE | build, run, test, proto, backfill, train, backtest |
| Backfill pipeline | DONE | `python/research/backfill.py` — 30-day historical data (34,564 rows) |
| Model training | DONE | `python/research/train_model.py` — CatBoost/XGBoost/LightGBM + Optuna + SHAP + calibration |
| Backtesting | DONE | `python/research/backtest.py` — threshold sweep, quarter-Kelly flat sizing, fee model |
| Directional strategy | DONE | `internal/strategy/directional.go` (~345 LOC) — gRPC signal every 3s, PostOnly/FOK, accuracy tracking |
| Market maker | DONE | `internal/strategy/market_maker.go` (~380 LOC) — fair value, 4-cent spread, inventory skew |
| Strategy router | DONE | `internal/strategy/router.go` (~60 LOC) — concurrent dispatcher, 60/40 allocation |
| Docker hardening | DONE | Non-root users, health checks, restart policies, pinned versions, shared network |
| Prometheus alerts | DONE | `deploy/alerts.yml` — 6 rules (daily loss, WS, latency, exposure, accuracy, drawdown) |
| Alertmanager + Telegram | DONE | `deploy/alertmanager.yml` — Telegram bot notifications |
| Grafana dashboard | DONE | 11-panel dashboard with auto-provisioned datasource |

---

## ~~Sprint 0: Repo Scaffolding + Config + WebSocket Connections~~ COMPLETE

All deliverables verified working: `make build && make run` connects to Binance, Polymarket, and Deribit. Graceful shutdown on Ctrl+C. Auto-reconnect on network drop.

---

## ~~Sprint 1: Event Bus + TA Engine + Position Manager + Risk + Metrics~~ COMPLETE

All deliverables verified: `make test` passes 15 unit tests (position + risk). Event bus wired to all WS clients. TA engine consuming Binance trades. Metrics server on :9090. `temporal_arb.go` deleted.

---

## ~~Sprint 2: Order Execution + Market Discovery~~ COMPLETE

All deliverables verified: `make test` passes 22 unit tests (position 10, risk 5, executor 7). Paper trading mode fills at limit price with 50-200ms simulated latency. CLOB client uses HMAC-SHA256 L2 auth. Market discovery polls Gamma API by slug (`btc-updown-5m-{bucket}`). Wired into main.go with graceful shutdown.

Key discoveries during implementation:
- `mtt-labs/poly-market-sdk` is 404/gone — built CLOB HTTP client directly
- Gamma API uses slug-based lookup, not tag-based (`?tag=btc-5-minute`) as originally planned
- Broke execution↔risk import cycle with `RiskChecker` interface in execution package
- Added `metrics.NewNoop()` for test-safe unregistered Prometheus metrics

### Prompt
```
In the poly-edge project, build the event bus, TA indicator engine, position manager, risk manager, and expand Prometheus metrics.

NOTE: metrics/metrics.go already exists with 6 basic metrics + HTTP server. Expand it — don't rewrite from scratch. Also delete internal/strategy/temporal_arb.go (not part of our strategy plan).

## 1. Event Bus (internal/eventbus/bus.go)

Currently an empty struct stub. Replace with full implementation.

Typed events as Go structs:
  BinanceBookUpdate{Bids []PriceLevel, Asks []PriceLevel, Timestamp int64}
  BinanceTrade{Price float64, Qty float64, IsBuyerMaker bool, Timestamp int64}
  BinanceLiquidation{Side string, Price float64, Qty float64, Timestamp int64}
  BinanceMarkPrice{MarkPrice float64, FundingRate float64, Timestamp int64}
  PolymarketBookUpdate{MarketID string, YesBestBid, YesBestAsk, NoBestBid, NoBestAsk float64}
  PolymarketPrice{Price float64, Timestamp int64}
  DeribitDVOL{Value float64, Timestamp int64}

Bus with Subscribe(eventType) <-chan Event and Publish(event).
Fan-out to multiple subscribers. Non-blocking publish: if channel full (buffer 100), drop + log warning. Thread-safe with mutex.

## 2. TA Engine (internal/ta/engine.go)

This file does NOT exist yet — create the internal/ta/ directory and engine.go.

Uses github.com/cinar/indicator for core math.

Maintains rolling 1-minute candles built from BinanceTrade events.
On each candle close computes:
  RSI(7), RSI(14), RSI(30)
  MACD(12,26,9) signal line value
  Stochastic %K(14)
  Bollinger %B (20, 2 sigma)
  ATR(14)
  EMA(9) vs EMA(21) difference, normalized by price

On each trade (sub-candle) computes:
  Momentum: price change over last 30s, 60s, 120s
  1H trend direction: sign of (price_now - price_60min_ago), values: 1.0, -1.0, 0.0

Method: GetIndicators() TASnapshot — returns latest values of all indicators.
Subscribe to BinanceTrade events from event bus.

## 3. Position Manager (internal/position/manager.go)

Currently an empty struct stub. Replace with full implementation.

MarketPosition: {MarketID, ConditionID, YesTokenID, NoTokenID string; YesShares, NoShares int; AvgCostYes, AvgCostNo float64; StrategyID string}
PortfolioState: {TotalBankroll, Deployed, Available, UnrealizedPnL, RealizedPnL, DailyPnL float64; DailyPnLResetAt time.Time}

Mutex-protected. Methods:
  RecordFill(marketID, side string, shares int, price float64, strategyID string)
  RecordResolution(marketID, winnerSide string) — compute realized PnL, free capital
  GetPosition(marketID string) MarketPosition
  GetPortfolio() PortfolioState
  CanAllocate(strategyID string, amount float64) bool

Capital allocation: directional 60%, market_maker 40% of BankrollUSDC.
Daily PnL resets at 00:00 UTC.
Write unit tests for fill recording, resolution PnL, and allocation limits.

## 4. Risk Manager (internal/risk/risk.go)

Currently an empty struct stub. Replace with full implementation.

PreTradeCheck(order OrderRequest, portfolio PortfolioState) (allowed bool, reason string)
Rules:
  a) Max position per market: 10% of bankroll
  b) Max total exposure: 60% of bankroll
  c) Daily loss limit: 2.5% of bankroll
  d) Single trade max: 2% of bankroll
Write unit tests for each rule.

## 5. Prometheus Metrics (internal/metrics/metrics.go)

EXISTING: Already has OrderLatency, OrdersPlaced, OrderErrors, OrdersCanceled, PositionSize, FillsReceived metrics + HTTP server on configurable port.

EXPAND to include these additional metrics:
  polyedge_orders_total (CounterVec: strategy, side, status) — replace OrdersPlaced
  polyedge_order_latency_seconds (HistogramVec: strategy) — replace OrderLatency
  polyedge_fill_rate (GaugeVec: strategy)
  polyedge_pnl_realized (GaugeVec: strategy)
  polyedge_pnl_unrealized (GaugeVec: strategy)
  polyedge_position_exposure (GaugeVec: strategy, side)
  polyedge_ws_connected (GaugeVec: source)
  polyedge_drawdown_current (Gauge)
  polyedge_bankroll_available (Gauge)
  polyedge_directional_accuracy (Gauge)
  polyedge_directional_confidence_avg (Gauge)
  polyedge_mm_inventory (GaugeVec: side)
  polyedge_mm_spread_bps (Gauge)
Keep the existing Serve() pattern.

## 6. Wire Together

WS handlers publish events to bus. TA engine subscribes to BinanceTrade.
WS connection status updates ws_connected gauge. Start metrics HTTP server.
```

### Done When
`make test` passes position + risk unit tests. `curl localhost:9090/metrics` returns live metrics. TA engine logs RSI/MACD values verifiable against TradingView.

---

### Prompt (preserved for reference)
```
In poly-edge, build the order execution layer for Polymarket.

NOTE: internal/execution/order_types.go already has order model definitions (Side, OrderType, OrderRequest with validation, OrderStatus, OrderResult, FillEvent). Review it and extend if needed — don't rewrite.

## 1. Executor (internal/execution/executor.go)

Currently an empty struct stub. Replace with full implementation.

Uses github.com/Polymarket/go-order-utils for EIP-712 signing.
Uses github.com/mtt-labs/poly-market-sdk for CLOB API.
Initialize with config (private key, API creds, chain ID 137).

PlaceOrder(req OrderRequest) (OrderResponse, error):
  1. Run risk.PreTradeCheck — reject if not allowed
  2. Build order with go-order-utils (EIP-712 typed data)
  3. Submit via poly-market-sdk
  4. Record latency metric, increment orders counter
  5. On success: call positionManager.RecordFill
  6. Return response

CancelOrder(orderID string) error
CancelAllForMarket(conditionID string) error
PlaceBatch(orders []OrderRequest) ([]OrderResponse, error) — batch up to 15

Paper trading mode (config.PaperTrade == true):
  PlaceOrder logs full order JSON but does NOT submit to Polymarket
  Simulates fill at limit price after random 50-200ms delay
  Calls positionManager.RecordFill with simulated data
  Returns synthetic OrderResponse with Status "PAPER_FILLED"
  All downstream code works identically in paper and live mode.

## 2. Market Discovery (internal/execution/market_discovery.go)

This file does NOT exist yet — create it.

DiscoverCurrentMarket() (MarketInfo, error):
  GET https://gamma-api.polymarket.com/events?tag=btc-5-minute&active=true
  Parse: MarketInfo{EventID, ConditionID, YesTokenID, NoTokenID, StartTime, EndTime}

DiscoverUpcomingMarket() — next market that hasn't started.
Poll every 60 seconds. On new market: log it, subscribe CLOB WS to its order book.

Wire into main.go. Log discovered markets.
```

### Done When
Paper mode: place order -> logged with correct structure -> position manager updated -> Prometheus counters incremented. Market discovery logs current/upcoming 5-min BTC markets.

---

## ~~Sprint 3: gRPC Bridge + Python Feature Engineering~~ COMPLETE

All deliverables verified: Go SignalClient assembles 23-field MarketState from cached event bus data + TA indicators, calls Python sidecar over UDS. Python computes 26 features (10 order flow, 12 TA pass-through, 4 derived). CatBoost wrapper returns "hold" until model.cbm exists. Round-trip tested with dummy MarketState.

Key decisions:
- Go is gRPC **client**, Python is gRPC **server** (Go assembles data, Python does inference)
- Rate limited to max 2 calls/sec to avoid overloading sidecar
- `WaitForSidecar()` runs in background goroutine — Go starts fine without Python running
- Generated stubs use `signals_pb2` import path; server.py adds proto/ to sys.path
- Project venv at `.venv/` (shared by sidecar + research), proto generation via `.venv/bin/python`

### Prompt (preserved for reference)
```
In poly-edge, build the gRPC bridge between Go and Python and the Python feature pipeline.

NOTE: internal/grpc/proto/signals.proto exists but has a minimal schema. Replace it with the full spec below.

## 1. Protobuf (internal/grpc/proto/signals.proto)

Replace the existing proto with:

syntax = "proto3";
package polyedge;

service SignalService {
  rpc GetSignal (MarketState) returns (Signal);
}

message MarketState {
  double btc_price = 1;
  double window_open_price = 2;
  double seconds_remaining = 3;
  double polymarket_yes_price = 4;
  double polymarket_no_price = 5;
  double dvol = 6;
  repeated double binance_bids = 7;     // flattened: [price, qty, price, qty, ...]
  repeated double binance_asks = 8;
  repeated AggTrade recent_trades = 9;
  repeated Liquidation recent_liquidations = 10;
  double funding_rate = 11;
  // Pre-computed TA from Go
  double rsi_7 = 12; double rsi_14 = 13; double rsi_30 = 14;
  double macd_signal = 15; double stoch_k = 16;
  double momentum_30s = 17; double momentum_60s = 18; double momentum_120s = 19;
  double ema_9_vs_21 = 20; double atr_14 = 21;
  double bollinger_pct_b = 22; double hourly_trend = 23;
}

message AggTrade {
  double price = 1; double quantity = 2; bool is_buyer_maker = 3; int64 timestamp_ms = 4;
}

message Liquidation {
  string side = 1; double quantity = 2; double price = 3; int64 timestamp_ms = 4;
}

message Signal {
  string action = 1;          // "buy_yes", "buy_no", "hold"
  double confidence = 2;      // 0.0-1.0
  double suggested_price = 3;
  double suggested_size = 4;
}

Generate Go and Python stubs. Add to Makefile `proto` target.

## 2. Go gRPC Server (internal/grpc/server.go)

Currently an empty struct stub. Replace with full implementation.

Listen on Unix Domain Socket at config.GrpcSocketPath.
On each GetSignal call: assemble MarketState from latest WS data + ta.GetIndicators().
Throttle: max 2 gRPC calls/sec to Python.

## 3. Python Sidecar

All Python sidecar files are currently 1-line docstrings. Replace with full implementations.

python/sidecar/features.py:
Compute order flow features from MarketState:
  order_flow_imbalance_L5: sum(bid_qty L1-L5) / sum(all_qty L1-L5)
  order_flow_imbalance_L10: same for levels 5-10
  normalized_spread: (best_ask - best_bid) / midprice
  vwap_to_mid_bid, vwap_to_mid_ask
  trade_aggressor_ratio_30s/60s/120s: buy_volume / total_volume
  cvd_60s: cumulative (buy_vol - sell_vol)
  liquidation_imbalance_60s: long_liq_vol - short_liq_vol

Combine with pre-computed TA from MarketState (rsi_7, rsi_14, rsi_30, macd_signal, stoch_k, momentum_30s/60s/120s, ema_9_vs_21, atr_14, bollinger_pct_b, hourly_trend).

Add derived features:
  price_vs_open: (btc_price - window_open_price) / window_open_price
  time_decay: seconds_remaining / 300.0
  dvol_level: raw DVOL
  mean_reversion_signal: 1.0 if price moved > 0.3% from open (captures BTC's negative autocorrelation at 5-min)

Return all 25 features as dict.

python/sidecar/model.py:
Load CatBoost model from model.cbm on startup. If no model: return hold with confidence 0.0.
predict(features_dict) -> (action, confidence)

python/sidecar/server.py:
gRPC server. Receive MarketState -> features.py -> model.py -> return Signal.

Wire into main.go: start gRPC server in goroutine.
```

### Done When
Go sends MarketState, Python receives, computes features, returns Signal. Feature values look reasonable (OFI 0-1, RSI 0-100). Round-trip < 10ms.

---

## ~~Sprint 4: Training Data Collection~~ COMPLETE

### Prompt
```
In poly-edge, build the training data collection pipeline. This runs alongside the live system recording feature snapshots + outcomes for model training.

All Python research files are currently 1-line docstrings. Replace with full implementations.

## 1. Historical Backfill (python/research/backfill.py) — NEW

Build a backfill script using Binance REST API to generate training data from historical aggTrades.
This lets us train an initial model within days instead of waiting 7-14 days for live data.

- Fetch historical aggTrades from Binance Futures REST API (GET /fapi/v1/aggTrades)
- Reconstruct 5-min windows aligned to boundaries (00:00, 00:05, 00:10... UTC)
- For each window: build 1-min candles, compute TA indicators, compute order flow features
- Available features from historical data (~12 of 26): all TA indicators + momentum + trade
  aggressor ratios + CVD. NOT available: live order book (OFI, spread, VWAP), Polymarket prices,
  DVOL, liquidations.
- Label: close >= open = 1, else 0
- Output: Parquet with same schema as live collector (missing features as NaN)
- Target: 30+ days of historical data (~8,640 windows)
- Run: python research/backfill.py --days 30 --output-dir data/training/

## 2. Live Collector (python/research/collect_training_data.py)

- Connect to Binance Futures WS (btcusdt@depth20@100ms, @aggTrade, @forceOrder) and Polymarket RTDS
- Align to 5-min window boundaries (00:00, 00:05, 00:10... UTC)
- Every window, record snapshots at T+30s, T+60s, T+90s, T+120s, T+180s, T+240s
- Each snapshot: all 26 features + btc_price + polymarket YES/NO prices + window metadata
- At window close (T+300s): label outcome (close >= open = 1, else 0)
- Save to Parquet: one file per day, append outcome after each window resolves
- Run as standalone: python collect_training_data.py --output-dir data/training/
- Target: minimum 7 days (2,016 windows x 6 snapshots = 12,096 rows)

## 3. Label Validation (python/research/label_outcomes.py)

- Read Parquet files, fill missing outcomes from Binance klines API
- Validate: no NaN outcomes, no future data leakage
- Output clean labeled dataset

START BOTH: run backfill immediately for historical data, start live collector in parallel
for the full 26-feature set. Train initial model on backfill data while live data accumulates.
```

### Done When
Backfill produces 30 days of historical data. Live collector captures ~12 windows/hour with correct features. Both output compatible Parquet schemas.

---

## ~~Sprint 5: Model Training + Backtesting~~ COMPLETE

### Prompt
```
In poly-edge, build model training and backtesting.

## 1. Training (python/research/train_model.py)

Load labeled Parquet data. Use all 26 features:
  order_flow_imbalance_L5, order_flow_imbalance_L10, normalized_spread,
  vwap_to_mid_bid, vwap_to_mid_ask,
  trade_aggressor_ratio_30s/60s/120s, cvd_60s, liquidation_imbalance_60s,
  rsi_7, rsi_14, rsi_30, macd_signal, stoch_k,
  momentum_30s/60s/120s, ema_9_vs_21, atr_14, bollinger_pct_b,
  hourly_trend, price_vs_open, time_decay, dvol_level, mean_reversion_signal

For backfill-only data (missing OFI, spread, VWAP, DVOL, liquidations):
  Train on available ~12 features first. Retrain with full 26 once live data accumulates.

TIME-BASED SPLIT ONLY: first 80% train, last 20% test. NEVER random split.

Train three models: CatBoost, XGBoost, LightGBM. Compare.
Hyperparameters via Optuna:
  learning_rate: [0.01, 0.03, 0.05, 0.1]
  depth: [3, 4, 5, 6]
  l2_leaf_reg: [1, 3, 5, 7]
  iterations: [200, 500, 1000]
Use TimeSeriesSplit cross-validation (expanding window, not k-fold).

Evaluation:
  Accuracy (target 53-58% — above 60% means data leakage)
  AUC-ROC, Brier score, log loss
  Accuracy by confidence bucket (top 10%, 20%, 50%)
  Accuracy by time of day (Asian/European/US session)
  Accuracy by volatility regime (high/low DVOL)

## 2. Calibration — NEW

Post-training probability calibration to ensure model outputs are well-calibrated:
  - Apply Platt scaling (logistic) and isotonic regression to raw model probabilities
  - Compare calibration curves (reliability diagram) before/after
  - Pick whichever method produces lower Brier score on validation set
  - Save calibrator alongside model (model.cbm + calibrator.pkl)

## 3. Regime-Specific Models — NEW

Train separate models for high-DVOL and low-DVOL regimes:
  - Split training data by DVOL median into high-vol and low-vol subsets
  - Train separate CatBoost models for each regime
  - Compare regime-specific accuracy vs single model
  - If regime models outperform: save as model_high_dvol.cbm + model_low_dvol.cbm
    and update model.py to dispatch based on current DVOL

## 4. Backtesting (python/research/backtest.py)

Simulate Polymarket trading on test set:
  Model predicts at T+60s snapshot per window.
  Trade only when confidence > threshold.
  Sweep thresholds: 0.52, 0.55, 0.58, 0.60, 0.62, 0.65

Fee model:
  PostOnly (default): 0%
  FOK taker (confidence > 0.65 & time < 120s): shares x 0.0025 x p x (1-p)

Position sizing: quarter-Kelly
  edge = model_probability - market_price
  kelly = edge / (1 - market_price)
  size = 0.25 x kelly x available_capital

Output: cumulative PnL curve (PNG), Sharpe, max drawdown, win rate, trades/day, PnL by threshold.
Include baseline: random 50% predictor with same sizing.
```

### Done When
Model at 53-58% test accuracy. SHAP shows feature importance. Calibration curve shows improvement. Backtest positive after fees at best threshold. All output files generated.

---

## ~~Sprint 6: Directional Strategy + Market Making~~ COMPLETE

### Prompt
```
In poly-edge, integrate the trained model as the directional strategy and build the market maker.

All strategy files are currently empty struct stubs. Replace with full implementations.

## 1. Directional Strategy (internal/strategy/directional.go)

Subscribe to PolymarketBookUpdate events.
Every 3 seconds during a 5-min window:
  Call Python sidecar via gRPC with current MarketState
  Receive Signal (action, confidence)

Trading logic:
  If confidence > 0.58 AND action == "buy_yes": PostOnly buy YES at best_ask - $0.01
  If confidence > 0.58 AND action == "buy_no": PostOnly buy NO at best_ask - $0.01
  If confidence > 0.65 AND seconds_remaining < 120: FOK taker order instead
  One position per window max. Hold to resolution.

Position sizing: quarter-Kelly
  edge = model_confidence - market_price
  kelly_fraction = edge / (1 - market_price)
  size = 0.25 x kelly_fraction x available_capital
  Capped by risk.PreTradeCheck and 60% directional allocation limit

Live accuracy tracking:
  Rolling 100-window accuracy counter
  If accuracy drops below 50% for 50+ windows: pause strategy, log alert
  Update polyedge_directional_accuracy gauge
  Update polyedge_directional_confidence_avg gauge

## 2. Market Making Strategy (internal/strategy/market_maker.go)

Subscribe to PolymarketBookUpdate and PolymarketPrice events.

Fair value estimation (two modes):
  Primary: use calibrated directional model output as fair_yes probability
    When model confidence > 0.55, fair_yes = calibrated P(up) from sidecar
    This is better than the linear formula because it uses the full feature set
  Fallback (when model unavailable or confidence < 0.55):
    fair_yes = 0.50 + k x (current_chainlink_price - window_open_price) / window_open_price
    k = 5.0 (sensitivity, configurable, tune later)

Quoting:
  Spread: 4 cents wide around fair_yes. E.g. fair=0.52 -> bid YES 0.50, ask YES 0.54
  ALL orders PostOnly (0% fee guaranteed)
  Quote both YES and NO sides. Max 4 resting orders (2 per side).

Refresh triggers:
  a) Price moves > 1.5 cents from last quote midpoint
  b) Every 30 seconds as heartbeat
  On refresh: cancel all resting orders, place new batch (PlaceBatch for efficiency)

Inventory management:
  net_inventory = yes_shares - no_shares
  If net_inventory > 0: lower YES bid by 1 cent, raise NO bid by 1 cent
  If net_inventory < 0: opposite
  Max inventory: 200 shares per side (configurable)
  If exceeded: stop quoting the full side, only reducing orders

Integration with directional:
  When directional confidence > 0.65: skew MM quotes in same direction
  When directional has active position: MM can still quote the same market (they complement)
  Capital allocation: 40% of bankroll to MM

Metrics:
  polyedge_mm_quotes_placed (counter)
  polyedge_mm_inventory{side} (gauge)
  polyedge_mm_spread_bps (gauge)

## 3. Strategy Router (internal/strategy/router.go)

Simple dispatcher. Both strategies run concurrently on same markets.
Directional = primary (gets first right of capital allocation).
MM = complement (uses remaining allocation).
Both check risk.PreTradeCheck before every order.

Wire both into main.go, subscribe to events.
Run in paper mode. Both should operate concurrently without capital conflicts.
```

### Done When
Paper mode: directional makes predictions with logged confidence, MM places/refreshes quotes. Both run concurrently. Prometheus shows all strategy metrics. No capital allocation conflicts.

All deliverables verified: `go build`, `go vet`, and all tests pass. Directional strategy polls sidecar every 3s, places PostOnly/FOK orders with quarter-Kelly sizing, tracks rolling 100-window accuracy with auto-pause. Market maker quotes 4-cent spread around model or Chainlink fair value, refreshes on 1.5-cent move or 30s heartbeat, manages inventory with skewing. Router runs both concurrently with 60/40 capital split. Wired into main.go with graceful shutdown.

---

## ~~Sprint 7: Deployment Hardening + Observability~~ COMPLETE

**Pre-existing work:**
- `deploy/Dockerfile.go` — multi-stage build already exists
- `deploy/Dockerfile.python` — exists but will fail (sidecar is empty until Sprint 3)
- `deploy/docker-compose.yml` — 4 services already configured (core, sidecar, prometheus, grafana)
- `deploy/prometheus.yml` — scrape config exists (5s interval, poly-edge:9090)
- `deploy/grafana/dashboards/trading.json` — exists

This sprint focuses on **hardening** the existing deploy stack and adding alerting, not building from scratch.

### Prompt
```
In poly-edge, harden the existing Docker deployment and add monitoring/alerting.

NOTE: deploy/ already has working Dockerfiles, docker-compose.yml, prometheus.yml, and a Grafana dashboard. Review and improve rather than rewriting.

## Prometheus Alert Rules

Add to deploy/prometheus.yml (or a separate alerts.yml):
  daily_loss_exceeded: polyedge_pnl_realized < -(bankroll x 0.025)
  ws_disconnected: polyedge_ws_connected == 0 for > 15s
  high_latency: histogram_quantile(0.99, polyedge_order_latency_seconds) > 0.5
  high_exposure: sum(polyedge_position_exposure) > bankroll x 0.6

## Grafana Dashboard

Review existing deploy/grafana/dashboards/trading.json and ensure it has panels for:
  Real-time PnL line chart (stacked: directional + market_maker)
  Order latency histogram (heatmap, p50/p95/p99)
  Fill rate per strategy (gauge)
  Position exposure by strategy and side (bar)
  WebSocket connection status (status map: green/red)
  Drawdown gauge (red above 2%)
  Bankroll available (single stat)
  Directional model accuracy (rolling 100-window line chart)
  Directional confidence distribution (histogram)
  MM inventory balance (line: yes vs no shares)
  Orders per minute by strategy (bar)

Auto-provision datasource and dashboard.

## Alerting

Telegram bot: TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in config.
Alerts for: daily PnL summary, risk limit breach, WS disconnect > 30s, kill switch triggered.
```

### Done When
`make docker-up` starts all 4 services. Grafana at :3000 shows live dashboard with data. Prometheus alerts configured. Telegram alerts fire on test trigger.

---

## Sprint 8: Paper Trading Validation (5–7 days, no code)

### Prompt
```
This is a validation sprint — no code unless bugs found.

Run full system in paper mode via docker-compose for minimum 5 days.

Daily review checklist (log in VALIDATION.md):

1. DIRECTIONAL:
   - Predictions made today?
   - Rolling accuracy? (target > 53%)
   - Average confidence of trades taken?
   - Confidence threshold filtering correctly?
   - Paper PnL for the day?
   - Accuracy by session (Asian/European/US)?

2. MARKET MAKER:
   - Quotes placed and filled?
   - Inventory balance (YES vs NO)?
   - Inventory skewing working?
   - Estimated rebate earnings?

3. SYSTEM HEALTH:
   - WS disconnects? Duration?
   - Order latency p50/p95/p99?
   - Risk limit breaches?
   - CPU/RAM on VPS?
   - Error logs?

4. PARAMETER TUNING (adjust based on observations):
   - Directional confidence threshold (default 0.58)
   - MM spread width (default 4 cents)
   - MM inventory skew factor
   - MM fair value sensitivity k (default 5.0)
   - Risk limits

PASS CRITERIA:
   - 5+ consecutive days positive total paper PnL
   - No unhandled crashes
   - Directional accuracy > 52% over 500+ predictions
   - All WS connections stable (< 5 disconnects/day)
```

### Done When
VALIDATION.md documents 5+ days with metrics. Pass criteria met.

---

## Sprint 9: Go Live (1–2 days)

### Prompt
```
In poly-edge, transition to live trading with real capital.

## Phased deployment

Phase 1 (days 1-3): $200 USDC
  PAPER_TRADE=false, DEPLOYMENT_PHASE=1
  Risk limits: max $20/trade, $10 daily loss limit, $100 total exposure
  Monitor actively first 4 hours via Grafana

Phase 2 (days 4-7): $500 if Phase 1 profitable
  DEPLOYMENT_PHASE=2
  Risk limits: max $50/trade, $25 daily loss limit, $300 total exposure

Phase 3 (week 2+): full bankroll if Phase 2 profitable
  DEPLOYMENT_PHASE=3, standard risk limits

## Config

Add DEPLOYMENT_PHASE to config. Risk limits auto-adjust per phase.

## Kill Switch

3 consecutive daily losses: auto-pause all strategies, alert via Telegram
Manual: touch /tmp/polyedge-pause to pause, rm to resume
Graceful pause: cancel resting orders, stop new orders, hold existing to resolution

## Telegram Alerts

Bot token/chat ID via config.
Alerts: daily PnL summary, risk breach, WS disconnect > 30s, kill switch, phase transition.
Use github.com/go-telegram-bot-api/telegram-bot-api/v5

## Parallel Paper Mode

Second instance with PAPER_TRADE=true on different metrics port.
Compare paper vs live PnL daily (detect execution slippage).

## Daily Reconciliation

python/research/reconcile.py:
  Pull actual Polymarket positions via Data API
  Compare to internal position manager state
  Flag discrepancies
  Run as daily cron

Update README with live operations guide.
```

### Done When
Live trades on Phase 1 ($200). First 24 hours show real trades, real PnL. Telegram alerts working. Reconciliation confirms positions match.

---

## Timeline Summary

| Sprint | Duration | Cumulative | Status |
|--------|----------|------------|--------|
| ~~S0: Scaffolding + WS~~ | ~~3-4 days~~ | — | **COMPLETE** |
| ~~S1: Event Bus + TA + Position + Risk + Metrics~~ | ~~2–3 days~~ | — | **COMPLETE** |
| ~~S2: Order Execution + Discovery~~ | ~~2–3 days~~ | — | **COMPLETE** |
| ~~S3: gRPC + Python Features~~ | ~~3–4 days~~ | — | **COMPLETE** |
| S4: Data Collection + Backfill | 1–2 days + 7–14 days passive | Day 12 (setup) | **NEXT** |
| S5: Model Training + Calibration + Backtest | 3–4 days (after backfill) | Day 16 | |
| S6: Directional + MM Strategies | 3–4 days | Day 31 | |
| S7: Deployment Hardening + Observability | 1–2 days | Day 33 | |
| S8: Paper Validation | 5–7 days | Day 40 | |
| S9: Go Live | 1–2 days | Day 42 | |

**Critical path:** Start S4 data collection immediately after S3. It runs passively 7-14 days.

**Total: ~6 weeks** at 3-4 hours/day (saved ~1 week from Sprint 0 + partial Sprint 7 being done).
