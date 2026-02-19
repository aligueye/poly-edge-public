package strategy

import (
	"context"
	"math"
	"sync"
	"time"

	"github.com/ali/poly-edge/internal/execution"
	polygrpc "github.com/ali/poly-edge/internal/grpc"
	"github.com/ali/poly-edge/internal/metrics"
	"github.com/ali/poly-edge/internal/position"
	"github.com/rs/zerolog"
)

const (
	dirSignalInterval   = 3 * time.Second
	dirConfThreshold    = 0.58
	dirFOKThreshold     = 0.65
	dirFOKTimeRemaining = 120.0 // seconds
	dirKellyFraction    = 0.25
	dirMaxPosCap        = 0.10 // 10% of bankroll max per trade
	dirAccuracyWindow   = 100
	dirPauseThreshold   = 50 // pause after 50 windows below 50%
)

// Directional implements the directional betting strategy.
// It calls the Python sidecar every 3 seconds and places bets
// when model confidence exceeds the threshold.
type Directional struct {
	signal    *polygrpc.SignalClient
	executor  *execution.Executor
	discovery *execution.MarketDiscovery
	position  *position.Manager
	metrics   *metrics.Metrics
	bankroll  float64
	log       zerolog.Logger

	// State
	mu               sync.Mutex
	currentWindowID  string // condition ID of current window
	hasPositionInWin bool   // already traded this window
	betSide          string // "YES" or "NO" — side bet in current window
	paused           bool

	// Rolling accuracy tracking
	outcomes    []bool // ring buffer of win/loss results
	outcomeIdx  int
	outcomeFull bool
	correct     int
	total       int
	belowCount  int // consecutive windows below 50%

	// Confidence tracking
	confSum        float64
	confCount      int
	lastAction     string
	lastConfidence float64

	done chan struct{}
}

// NewDirectional creates a directional strategy.
func NewDirectional(
	signal *polygrpc.SignalClient,
	executor *execution.Executor,
	discovery *execution.MarketDiscovery,
	posManager *position.Manager,
	met *metrics.Metrics,
	bankroll float64,
	logger zerolog.Logger,
) *Directional {
	return &Directional{
		signal:    signal,
		executor:  executor,
		discovery: discovery,
		position:  posManager,
		metrics:   met,
		bankroll:  bankroll,
		log:       logger.With().Str("component", "directional").Logger(),
		outcomes:  make([]bool, dirAccuracyWindow),
		done:      make(chan struct{}),
	}
}

// Run starts the directional strategy loop. Blocks until Stop() is called.
func (d *Directional) Run() {
	d.log.Info().
		Float64("conf_threshold", dirConfThreshold).
		Float64("fok_threshold", dirFOKThreshold).
		Float64("kelly_fraction", dirKellyFraction).
		Msg("directional strategy started")

	ticker := time.NewTicker(dirSignalInterval)
	defer ticker.Stop()

	for {
		select {
		case <-d.done:
			return
		case <-ticker.C:
			d.tick()
		}
	}
}

// Stop signals the strategy to shut down.
func (d *Directional) Stop() {
	close(d.done)
}

// RecordResolution records whether a directional bet won or lost.
// Called by the router when a market resolves.
func (d *Directional) RecordResolution(marketID string, won bool) {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Update ring buffer
	if d.outcomeFull {
		// Remove the old value being overwritten
		if d.outcomes[d.outcomeIdx] {
			d.correct--
		}
		d.total--
	}

	d.outcomes[d.outcomeIdx] = won
	d.outcomeIdx = (d.outcomeIdx + 1) % dirAccuracyWindow
	if d.outcomeIdx == 0 {
		d.outcomeFull = true
	}

	if won {
		d.correct++
	}
	d.total++

	accuracy := float64(d.correct) / float64(d.total)
	d.metrics.DirectionalAccuracy.Set(accuracy)

	d.log.Info().
		Str("market", marketID).
		Bool("won", won).
		Float64("accuracy", accuracy).
		Int("total", d.total).
		Msg("resolution recorded")

	// Check for pause condition
	if d.total >= dirPauseThreshold {
		if accuracy < 0.50 {
			d.belowCount++
			if d.belowCount >= dirPauseThreshold && !d.paused {
				d.paused = true
				d.log.Warn().
					Float64("accuracy", accuracy).
					Int("below_count", d.belowCount).
					Msg("PAUSING directional strategy — accuracy below 50%")
			}
		} else {
			d.belowCount = 0
			if d.paused {
				d.paused = false
				d.log.Info().Float64("accuracy", accuracy).Msg("resuming directional strategy")
			}
		}
	}
}

// GetConfidence returns the latest signal confidence (for MM skewing).
func (d *Directional) GetConfidence() (action string, confidence float64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.lastAction, d.lastConfidence
}

// GetBetSide returns the side ("YES"/"NO") the directional strategy bet on
// for the given conditionID, or "" if no bet was placed in that window.
func (d *Directional) GetBetSide(conditionID string) string {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.currentWindowID == conditionID && d.hasPositionInWin {
		return d.betSide
	}
	return ""
}

// Stats returns session stats for the shutdown summary.
func (d *Directional) Stats() (total, correct int, avgConf float64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.confCount > 0 {
		avgConf = d.confSum / float64(d.confCount)
	}
	return d.total, d.correct, avgConf
}

func (d *Directional) tick() {
	d.mu.Lock()
	if d.paused {
		d.mu.Unlock()
		return
	}
	d.mu.Unlock()

	// Get current market
	market := d.discovery.GetCurrent()
	if market == nil || !market.AcceptingOrders {
		return
	}

	secsRemaining := time.Until(market.EndTime).Seconds()
	if secsRemaining <= 0 {
		return
	}

	d.mu.Lock()
	// Check if this is a new window
	if market.ConditionID != d.currentWindowID {
		d.currentWindowID = market.ConditionID
		d.hasPositionInWin = false
		d.betSide = ""
	}

	if d.hasPositionInWin {
		d.mu.Unlock()
		return
	}
	d.mu.Unlock()

	if !d.signal.Connected() {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	signal, err := d.signal.GetSignal(ctx)
	if err != nil {
		d.log.Debug().Err(err).Msg("signal call failed")
		return
	}

	action := signal.GetAction()
	confidence := signal.GetConfidence()

	// Cache for MM cross-read
	d.mu.Lock()
	d.lastAction = action
	d.lastConfidence = confidence
	d.confSum += confidence
	d.confCount++
	if d.confCount > 0 {
		d.metrics.DirectionalConfidence.Set(d.confSum / float64(d.confCount))
	}
	d.mu.Unlock()

	if action == "hold" || confidence < dirConfThreshold {
		d.log.Debug().
			Str("action", action).
			Float64("confidence", confidence).
			Msg("signal below threshold")
		return
	}

	// Determine order parameters
	var side execution.Side
	var tokenID string
	var marketPrice float64

	switch action {
	case "buy_yes":
		side = execution.Buy
		tokenID = market.UpTokenID
		marketPrice = market.UpPrice
	case "buy_no":
		side = execution.Buy
		tokenID = market.DownTokenID
		marketPrice = market.DownPrice
	default:
		return
	}

	if marketPrice <= 0 || marketPrice >= 1 {
		marketPrice = 0.50
	}

	// Quarter-Kelly position sizing
	edge := confidence - marketPrice
	if edge <= 0 {
		d.log.Debug().
			Float64("confidence", confidence).
			Float64("market_price", marketPrice).
			Msg("no edge")
		return
	}

	kellyFrac := edge / (1.0 - marketPrice)
	sizeFrac := dirKellyFraction * kellyFrac

	// Available capital for directional (60% allocation), accounting for resting order collateral
	dirDeployed := d.position.StrategyDeployed("directional")
	dirPending := d.executor.PendingCapital("directional")
	dirAlloc := d.bankroll * position.DirectionalAlloc
	available := dirAlloc - dirDeployed - dirPending
	if available <= 0 {
		d.log.Debug().Float64("deployed", dirDeployed).Msg("directional allocation exhausted")
		return
	}

	positionSize := sizeFrac * d.bankroll
	maxCap := d.bankroll * dirMaxPosCap
	positionSize = math.Min(positionSize, maxCap)
	positionSize = math.Min(positionSize, available)

	// Determine order type and price
	orderType := execution.OrderTypePostOnly
	price := marketPrice - 0.01
	if price < 0.01 {
		price = 0.01
	}

	if confidence > dirFOKThreshold && secsRemaining < dirFOKTimeRemaining {
		orderType = execution.OrderTypeFOK
		price = marketPrice
	}

	// Enforce minimum order size of 5 shares (Polymarket minimum)
	minPosition := 5.0 * price
	positionSize = math.Max(positionSize, minPosition)

	shares := positionSize / price

	if !d.position.CanAllocate("directional", price*shares) {
		d.log.Debug().Msg("position allocation check failed")
		return
	}

	req := execution.OrderRequest{
		StrategyID: "directional",
		TokenID:    tokenID,
		MarketID:   market.ConditionID,
		Side:       side,
		Price:      price,
		Size:       shares,
		OrderType:  orderType,
	}

	d.log.Info().
		Str("action", action).
		Float64("confidence", confidence).
		Str("order_type", string(orderType)).
		Float64("price", price).
		Float64("shares", shares).
		Float64("position_usd", positionSize).
		Float64("secs_remaining", secsRemaining).
		Msg("placing directional order")

	result, err := d.executor.PlaceOrder(req)
	if err != nil {
		d.log.Warn().Err(err).Msg("directional order failed")
		return
	}

	// Track which side we bet on for resolution accuracy
	betSide := "YES"
	if action == "buy_no" {
		betSide = "NO"
	}

	d.mu.Lock()
	d.hasPositionInWin = true
	d.betSide = betSide
	d.mu.Unlock()

	d.log.Info().
		Str("order_id", result.OrderID).
		Str("status", string(result.Status)).
		Float64("latency_ms", result.LatencyMs).
		Msg("directional order placed")
}
