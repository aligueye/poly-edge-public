package ta

import (
	"math"
	"sync"
	"time"

	"github.com/ali/poly-edge/internal/eventbus"
	"github.com/cinar/indicator/v2/helper"
	"github.com/cinar/indicator/v2/momentum"
	"github.com/cinar/indicator/v2/trend"
	"github.com/cinar/indicator/v2/volatility"
	"github.com/rs/zerolog"
)

const (
	maxCandles    = 120 // 2 hours of 1-min candles
	maxTrades     = 7200 // ~2 hours of trades at ~1/sec
	candleDuration = 60 * time.Second
)

// Candle is a 1-minute OHLCV bar built from BinanceTrade events.
type Candle struct {
	Open, High, Low, Close float64
	Volume                 float64
	Start                  time.Time
}

// TASnapshot holds the latest computed indicator values.
type TASnapshot struct {
	RSI7          float64
	RSI14         float64
	RSI30         float64
	MACDSignal    float64
	StochK        float64
	BollingerPctB float64
	ATR14         float64
	EMA9vsEMA21   float64 // (EMA9 - EMA21) / price, normalized
	Momentum30s   float64
	Momentum60s   float64
	Momentum120s  float64
	HourlyTrend   float64 // 1.0, -1.0, or 0.0
	Ready         bool    // true once enough candles exist for all indicators
}

type tradePoint struct {
	price float64
	ts    time.Time
}

// Engine computes technical indicators from Binance trade data.
type Engine struct {
	mu      sync.RWMutex
	candles []Candle
	current *Candle
	snap    TASnapshot
	trades  []tradePoint

	log  zerolog.Logger
	done chan struct{}
}

// NewEngine creates a TA engine. Call Run() to start consuming events.
func NewEngine(logger zerolog.Logger) *Engine {
	return &Engine{
		candles: make([]Candle, 0, maxCandles),
		trades:  make([]tradePoint, 0, maxTrades),
		log:     logger.With().Str("component", "ta").Logger(),
		done:    make(chan struct{}),
	}
}

// GetIndicators returns the latest snapshot (thread-safe copy).
func (e *Engine) GetIndicators() TASnapshot {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.snap
}

// Run subscribes to BinanceTrade events and computes indicators.
// Blocks until Stop() is called.
func (e *Engine) Run(bus *eventbus.Bus) {
	ch := bus.Subscribe(eventbus.EventBinanceTrade)
	e.log.Info().Msg("TA engine started")
	for {
		select {
		case <-e.done:
			return
		case evt := <-ch:
			trade := evt.(eventbus.BinanceTrade)
			e.processTrade(trade)
		}
	}
}

// Stop signals the engine to shut down.
func (e *Engine) Stop() {
	close(e.done)
}

func (e *Engine) processTrade(t eventbus.BinanceTrade) {
	ts := time.UnixMilli(t.Timestamp)
	price := t.Price
	qty := t.Qty

	e.mu.Lock()
	defer e.mu.Unlock()

	// Record trade point for momentum/trend
	e.trades = append(e.trades, tradePoint{price: price, ts: ts})
	if len(e.trades) > maxTrades {
		e.trades = e.trades[len(e.trades)-maxTrades:]
	}

	// Build 1-min candles
	if e.current == nil {
		e.current = &Candle{
			Open:   price,
			High:   price,
			Low:    price,
			Close:  price,
			Volume: qty,
			Start:  ts.Truncate(candleDuration),
		}
		return
	}

	// Check if this trade belongs to a new candle
	if ts.Sub(e.current.Start) >= candleDuration {
		// Close current candle
		e.candles = append(e.candles, *e.current)
		if len(e.candles) > maxCandles {
			e.candles = e.candles[len(e.candles)-maxCandles:]
		}

		// Recompute candle-based indicators
		e.computeCandle()

		// Start new candle
		e.current = &Candle{
			Open:   price,
			High:   price,
			Low:    price,
			Close:  price,
			Volume: qty,
			Start:  ts.Truncate(candleDuration),
		}
	} else {
		// Update current candle
		e.current.Close = price
		if price > e.current.High {
			e.current.High = price
		}
		if price < e.current.Low {
			e.current.Low = price
		}
		e.current.Volume += qty
	}

	// Update sub-candle indicators (momentum, hourly trend)
	e.computeTradeBased(price, ts)
}

func (e *Engine) computeCandle() {
	n := len(e.candles)
	if n < 31 { // Need at least 31 candles for RSI(30) + 1
		return
	}

	closings := make([]float64, n)
	highs := make([]float64, n)
	lows := make([]float64, n)
	for i, c := range e.candles {
		closings[i] = c.Close
		highs[i] = c.High
		lows[i] = c.Low
	}

	e.snap.RSI7 = chanLast(momentum.NewRsiWithPeriod[float64](7).Compute(helper.SliceToChan(closings)))
	e.snap.RSI14 = chanLast(momentum.NewRsiWithPeriod[float64](14).Compute(helper.SliceToChan(closings)))
	e.snap.RSI30 = chanLast(momentum.NewRsiWithPeriod[float64](30).Compute(helper.SliceToChan(closings)))

	_, macdsignal := trend.NewMacdWithPeriod[float64](12, 26, 9).Compute(helper.SliceToChan(closings))
	e.snap.MACDSignal = chanLast(macdsignal)

	stochK, _ := momentum.NewStochasticOscillator[float64]().Compute(
		helper.SliceToChan(highs),
		helper.SliceToChan(lows),
		helper.SliceToChan(closings),
	)
	e.snap.StochK = chanLast(stochK)

	e.snap.BollingerPctB = chanLast(volatility.NewPercentB[float64]().Compute(helper.SliceToChan(closings)))

	e.snap.ATR14 = chanLast(volatility.NewAtrWithPeriod[float64](14).Compute(
		helper.SliceToChan(highs),
		helper.SliceToChan(lows),
		helper.SliceToChan(closings),
	))

	ema9 := chanLast(trend.NewEmaWithPeriod[float64](9).Compute(helper.SliceToChan(closings)))
	ema21 := chanLast(trend.NewEmaWithPeriod[float64](21).Compute(helper.SliceToChan(closings)))
	lastPrice := closings[n-1]
	if lastPrice > 0 {
		e.snap.EMA9vsEMA21 = (ema9 - ema21) / lastPrice
	}

	e.snap.Ready = true

	e.log.Debug().
		Float64("rsi14", e.snap.RSI14).
		Float64("macd_signal", e.snap.MACDSignal).
		Float64("stoch_k", e.snap.StochK).
		Float64("bb_pctb", e.snap.BollingerPctB).
		Float64("atr14", e.snap.ATR14).
		Msg("indicators updated")
}

func (e *Engine) computeTradeBased(price float64, now time.Time) {
	e.snap.Momentum30s = e.momentum(now, 30*time.Second)
	e.snap.Momentum60s = e.momentum(now, 60*time.Second)
	e.snap.Momentum120s = e.momentum(now, 120*time.Second)

	// 1H trend: sign of (price_now - price_60min_ago)
	cutoff := now.Add(-60 * time.Minute)
	for i := len(e.trades) - 1; i >= 0; i-- {
		if e.trades[i].ts.Before(cutoff) || e.trades[i].ts.Equal(cutoff) {
			diff := price - e.trades[i].price
			if diff > 0 {
				e.snap.HourlyTrend = 1.0
			} else if diff < 0 {
				e.snap.HourlyTrend = -1.0
			} else {
				e.snap.HourlyTrend = 0.0
			}
			return
		}
	}
	e.snap.HourlyTrend = 0.0 // not enough data
}

func (e *Engine) momentum(now time.Time, window time.Duration) float64 {
	cutoff := now.Add(-window)
	for i := len(e.trades) - 1; i >= 0; i-- {
		if e.trades[i].ts.Before(cutoff) || e.trades[i].ts.Equal(cutoff) {
			old := e.trades[i].price
			if old > 0 {
				return (e.trades[len(e.trades)-1].price - old) / old
			}
			return 0
		}
	}
	return 0
}

// chanLast drains a channel and returns the last value.
func chanLast(ch <-chan float64) float64 {
	last := math.NaN()
	for v := range ch {
		last = v
	}
	return last
}
