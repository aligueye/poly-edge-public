package execution

import (
	"fmt"
	"math/rand"
	"strconv"
	"sync"
	"time"

	"github.com/ali/poly-edge/internal/metrics"
	"github.com/ali/poly-edge/internal/position"
	"github.com/rs/zerolog"
)

// RiskChecker is satisfied by risk.Engine. Defined here to avoid an import cycle
// (risk imports execution for OrderRequest).
type RiskChecker interface {
	PreTradeCheck(order OrderRequest, portfolio position.PortfolioState) (bool, string)
}

// Executor handles order submission and lifecycle management.
// In paper mode, it simulates fills without hitting the Polymarket CLOB.
type Executor struct {
	risk     RiskChecker
	position *position.Manager
	metrics  *metrics.Metrics
	clob     *ClobClient
	log      zerolog.Logger

	paperTrade bool

	mu           sync.Mutex
	activeOrders map[string]OrderResult // orderID -> result

	onFill func(FillEvent) // callback fired when a resting order fills
	stopCh chan struct{}
}

// NewExecutor creates an executor. In paper mode, no CLOB client is required.
func NewExecutor(
	riskEngine RiskChecker,
	posManager *position.Manager,
	met *metrics.Metrics,
	clob *ClobClient,
	paperTrade bool,
	logger zerolog.Logger,
) *Executor {
	return &Executor{
		risk:         riskEngine,
		position:     posManager,
		metrics:      met,
		clob:         clob,
		paperTrade:   paperTrade,
		log:          logger.With().Str("component", "executor").Logger(),
		activeOrders: make(map[string]OrderResult),
		stopCh:       make(chan struct{}),
	}
}

// Start performs startup reconciliation (cancels orphaned orders) and
// launches the fill polling loop for resting orders.
func (e *Executor) Start() {
	// P2: Cancel all resting orders from a previous session
	if !e.paperTrade && e.clob != nil {
		e.log.Info().Msg("cancelling all resting orders from previous session")
		if err := e.clob.CancelAll(); err != nil {
			e.log.Warn().Err(err).Msg("startup cancel-all failed")
		}
	}

	// Launch fill poller for live mode
	if !e.paperTrade && e.clob != nil {
		go e.pollFillsLoop()
	}
}

// Stop shuts down the fill polling loop.
func (e *Executor) Stop() {
	close(e.stopCh)
}

// OnFill registers a callback invoked when a resting order is filled.
func (e *Executor) OnFill(fn func(FillEvent)) {
	e.onFill = fn
}

// pollFillsLoop polls the CLOB API every 2s to detect fills on resting orders.
func (e *Executor) pollFillsLoop() {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-e.stopCh:
			return
		case <-ticker.C:
			e.pollFills()
		}
	}
}

func (e *Executor) pollFills() {
	// Snapshot active order IDs under lock
	e.mu.Lock()
	orderIDs := make([]string, 0, len(e.activeOrders))
	for id := range e.activeOrders {
		orderIDs = append(orderIDs, id)
	}
	e.mu.Unlock()

	if len(orderIDs) == 0 {
		return
	}

	for _, oid := range orderIDs {
		resp, err := e.clob.GetOrderStatus(oid)
		if err != nil {
			e.log.Debug().Err(err).Str("order_id", oid).Msg("poll order status failed")
			continue
		}

		status := OrderStatus(resp.Status)

		switch status {
		case StatusMatched:
			e.mu.Lock()
			tracked, exists := e.activeOrders[oid]
			if exists {
				delete(e.activeOrders, oid)
			}
			e.mu.Unlock()

			if !exists {
				continue
			}

			// Parse fill details from CLOB response
			sizeMatched, _ := strconv.ParseFloat(resp.SizeMatched, 64)
			price, _ := strconv.ParseFloat(resp.Price, 64)
			if sizeMatched <= 0 {
				sizeMatched = tracked.Request.Size
			}
			if price <= 0 {
				price = tracked.Request.Price
			}

			side := "YES"
			if tracked.Request.Side == Sell {
				side = "NO"
			}

			e.position.RecordFill(tracked.Request.MarketID, side, sizeMatched, price, tracked.Request.StrategyID)
			e.metrics.OrdersTotal.WithLabelValues(tracked.Request.StrategyID, tracked.Request.Side.String(), "filled").Inc()

			e.log.Info().
				Str("order_id", oid).
				Str("strategy", tracked.Request.StrategyID).
				Str("side", side).
				Float64("size", sizeMatched).
				Float64("price", price).
				Msg("resting order filled")

			if e.onFill != nil {
				e.onFill(FillEvent{
					OrderID:   oid,
					TokenID:   tracked.Request.TokenID,
					MarketID:  tracked.Request.MarketID,
					Side:      tracked.Request.Side,
					Price:     price,
					Size:      sizeMatched,
					Status:    "MATCHED",
					Timestamp: time.Now(),
				})
			}

		case StatusCanceled:
			e.mu.Lock()
			delete(e.activeOrders, oid)
			e.mu.Unlock()
			e.log.Debug().Str("order_id", oid).Msg("resting order canceled externally")
		}
	}
}

// PlaceOrder validates and submits a single order.
// In paper mode, simulates a fill at the limit price after a short delay.
func (e *Executor) PlaceOrder(req OrderRequest) (OrderResult, error) {
	start := time.Now()

	if err := req.Validate(); err != nil {
		e.metrics.OrderErrors.WithLabelValues("validation").Inc()
		return OrderResult{
			Status:   StatusFailed,
			ErrorMsg: err.Error(),
			Request:  req,
		}, err
	}

	// Risk check
	portfolio := e.position.GetPortfolio()
	allowed, reason := e.risk.PreTradeCheck(req, portfolio)
	if !allowed {
		e.metrics.OrderErrors.WithLabelValues("risk_rejected").Inc()
		e.log.Warn().
			Str("strategy", req.StrategyID).
			Str("reason", reason).
			Float64("price", req.Price).
			Float64("size", req.Size).
			Msg("order rejected by risk")
		return OrderResult{
			Status:   StatusFailed,
			ErrorMsg: reason,
			Request:  req,
		}, fmt.Errorf("risk check failed: %s", reason)
	}

	if e.paperTrade {
		return e.paperFill(req, start)
	}

	return e.livePlaceOrder(req, start)
}

// PendingCapital returns the total capital locked in resting orders for a strategy.
func (e *Executor) PendingCapital(strategyID string) float64 {
	e.mu.Lock()
	defer e.mu.Unlock()

	var total float64
	for _, res := range e.activeOrders {
		if res.Request.StrategyID == strategyID {
			total += res.Request.Price * res.Request.Size
		}
	}
	return total
}

// CancelOrder cancels a single resting order by ID.
func (e *Executor) CancelOrder(orderID string) error {
	if e.paperTrade {
		e.mu.Lock()
		delete(e.activeOrders, orderID)
		e.mu.Unlock()
		e.metrics.OrdersCanceled.Inc()
		e.log.Info().Str("order_id", orderID).Msg("paper cancel")
		return nil
	}

	if e.clob == nil {
		return fmt.Errorf("clob client not configured")
	}
	if err := e.clob.CancelOrder(orderID); err != nil {
		e.metrics.OrderErrors.WithLabelValues("cancel_failed").Inc()
		return err
	}

	e.mu.Lock()
	delete(e.activeOrders, orderID)
	e.mu.Unlock()
	e.metrics.OrdersCanceled.Inc()
	return nil
}

// CancelAllForMarket cancels all resting orders for a given condition ID.
func (e *Executor) CancelAllForMarket(conditionID string) error {
	if e.paperTrade {
		e.mu.Lock()
		var canceled int
		for id, res := range e.activeOrders {
			if res.Request.MarketID == conditionID {
				delete(e.activeOrders, id)
				canceled++
			}
		}
		e.mu.Unlock()
		e.metrics.OrdersCanceled.Add(float64(canceled))
		e.log.Info().Str("market", conditionID).Int("canceled", canceled).Msg("paper cancel market")
		return nil
	}

	if e.clob == nil {
		return fmt.Errorf("clob client not configured")
	}
	if err := e.clob.CancelMarketOrders(conditionID); err != nil {
		return err
	}
	// Remove cancelled orders from active tracking
	e.mu.Lock()
	for id, res := range e.activeOrders {
		if res.Request.MarketID == conditionID {
			delete(e.activeOrders, id)
		}
	}
	e.mu.Unlock()
	return nil
}

// PlaceBatch submits up to 15 orders. Returns results for each.
func (e *Executor) PlaceBatch(orders []OrderRequest) ([]OrderResult, error) {
	if len(orders) > 15 {
		return nil, fmt.Errorf("batch size %d exceeds max 15", len(orders))
	}

	results := make([]OrderResult, len(orders))
	for i, req := range orders {
		res, err := e.PlaceOrder(req)
		if err != nil {
			e.log.Warn().Err(err).Int("index", i).Msg("batch order failed")
		}
		results[i] = res
	}
	return results, nil
}

// paperFill simulates a fill at the limit price after a random 50-200ms delay.
func (e *Executor) paperFill(req OrderRequest, start time.Time) (OrderResult, error) {
	delay := time.Duration(50+rand.Intn(150)) * time.Millisecond
	time.Sleep(delay)

	latency := time.Since(start).Seconds()
	orderID := fmt.Sprintf("paper-%d", time.Now().UnixNano())

	// Determine side string for position manager
	side := "YES"
	if req.Side == Sell {
		side = "NO"
	}

	// Record the fill in position manager
	e.position.RecordFill(req.MarketID, side, req.Size, req.Price, req.StrategyID)

	// Update metrics
	e.metrics.OrdersTotal.WithLabelValues(req.StrategyID, req.Side.String(), "paper_filled").Inc()
	e.metrics.OrderLatency.WithLabelValues(req.StrategyID).Observe(latency)

	result := OrderResult{
		OrderID:     orderID,
		Status:      StatusPaper,
		SubmittedAt: start,
		LatencyMs:   latency * 1000,
		Request:     req,
	}

	e.log.Info().
		Str("order_id", orderID).
		Str("strategy", req.StrategyID).
		Str("side", req.Side.String()).
		Float64("price", req.Price).
		Float64("size", req.Size).
		Float64("latency_ms", result.LatencyMs).
		Msg("paper fill")

	return result, nil
}

// livePlaceOrder builds, signs, and submits an order to the Polymarket CLOB.
func (e *Executor) livePlaceOrder(req OrderRequest, start time.Time) (OrderResult, error) {
	if e.clob == nil {
		return OrderResult{
			Status:   StatusFailed,
			ErrorMsg: "clob client not configured",
			Request:  req,
		}, fmt.Errorf("clob client not configured")
	}

	resp, err := e.clob.PlaceOrder(req)
	latency := time.Since(start).Seconds()

	if err != nil {
		e.metrics.OrderErrors.WithLabelValues("submit_failed").Inc()
		e.metrics.OrderLatency.WithLabelValues(req.StrategyID).Observe(latency)
		return OrderResult{
			Status:      StatusFailed,
			ErrorMsg:    err.Error(),
			SubmittedAt: start,
			LatencyMs:   latency * 1000,
			Request:     req,
		}, err
	}

	status := OrderStatus(resp.Status)
	e.metrics.OrdersTotal.WithLabelValues(req.StrategyID, req.Side.String(), resp.Status).Inc()
	e.metrics.OrderLatency.WithLabelValues(req.StrategyID).Observe(latency)

	result := OrderResult{
		OrderID:     resp.OrderID,
		Status:      status,
		SubmittedAt: start,
		LatencyMs:   latency * 1000,
		Request:     req,
	}

	// If immediately matched, record fill
	if status == StatusMatched {
		side := "YES"
		if req.Side == Sell {
			side = "NO"
		}
		e.position.RecordFill(req.MarketID, side, req.Size, req.Price, req.StrategyID)
	}

	// Track resting orders
	if status == StatusLive {
		e.mu.Lock()
		e.activeOrders[resp.OrderID] = result
		e.mu.Unlock()
	}

	e.log.Info().
		Str("order_id", resp.OrderID).
		Str("status", resp.Status).
		Str("strategy", req.StrategyID).
		Float64("latency_ms", result.LatencyMs).
		Msg("order placed")

	return result, nil
}
