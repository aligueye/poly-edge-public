package strategy

import (
	"math"
	"sync"
	"time"

	"github.com/ali/poly-edge/internal/eventbus"
	"github.com/ali/poly-edge/internal/execution"
	"github.com/ali/poly-edge/internal/metrics"
	"github.com/ali/poly-edge/internal/position"
	"github.com/rs/zerolog"
)

const (
	mmSpreadCents      = 0.04 // 4-cent total spread
	mmHalfSpread       = mmSpreadCents / 2.0
	mmRefreshPriceMove = 0.015 // refresh on 1.5-cent price move
	mmRefreshInterval  = 30 * time.Second
	mmMaxInventory     = 200.0 // max shares per side
	mmFallbackK        = 5.0   // sensitivity for linear fair value
	mmSkewThreshold    = 0.65  // directional confidence for MM skewing
	mmSkewCents        = 0.01  // 1-cent skew adjustment
	mmMaxOrders        = 4     // max resting orders (2 per side)
)

// MarketMaker implements the market-making strategy.
// It quotes both YES and NO sides with a spread around fair value.
type MarketMaker struct {
	directional *Directional
	executor    *execution.Executor
	discovery   *execution.MarketDiscovery
	position    *position.Manager
	metrics     *metrics.Metrics
	bus         *eventbus.Bus
	bankroll    float64
	log         zerolog.Logger

	// State
	mu              sync.Mutex
	lastQuoteMid    float64
	lastRefreshTime time.Time
	currentWindowID string
	restingOrders   []string // order IDs of resting quotes

	// Cached from event bus
	chainlinkPrice float64
	windowOpen     float64

	refreshCh chan refreshRequest // async order placement
	done      chan struct{}
}

type refreshRequest struct {
	market  *execution.MarketInfo
	fairYes float64
}

// NewMarketMaker creates a market making strategy.
func NewMarketMaker(
	directional *Directional,
	executor *execution.Executor,
	discovery *execution.MarketDiscovery,
	posManager *position.Manager,
	met *metrics.Metrics,
	bus *eventbus.Bus,
	bankroll float64,
	logger zerolog.Logger,
) *MarketMaker {
	return &MarketMaker{
		directional: directional,
		executor:    executor,
		discovery:   discovery,
		position:    posManager,
		metrics:     met,
		bus:         bus,
		bankroll:    bankroll,
		log:         logger.With().Str("component", "market_maker").Logger(),
		refreshCh:   make(chan refreshRequest, 1),
		done:        make(chan struct{}),
	}
}

// Run starts the market maker loop. Blocks until Stop() is called.
func (mm *MarketMaker) Run() {
	mm.log.Info().
		Float64("spread_cents", mmSpreadCents).
		Float64("max_inventory", mmMaxInventory).
		Float64("refresh_move", mmRefreshPriceMove).
		Msg("market maker started")

	// Launch async refresh worker — CLOB I/O happens here, not in event loop
	go mm.refreshWorker()

	// Subscribe to price feed for Chainlink BTC/USD
	priceCh := mm.bus.Subscribe(eventbus.EventPolymarketPrice)

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-mm.done:
			return
		case evt := <-priceCh:
			price := evt.(eventbus.PolymarketPrice)
			mm.mu.Lock()
			mm.chainlinkPrice = price.Price
			mm.mu.Unlock()
		case <-ticker.C:
			// Drain all pending price events to get latest before ticking
			for drained := true; drained; {
				select {
				case evt := <-priceCh:
					price := evt.(eventbus.PolymarketPrice)
					mm.mu.Lock()
					mm.chainlinkPrice = price.Price
					mm.mu.Unlock()
				default:
					drained = false
				}
			}
			mm.tick()
		}
	}
}

// refreshWorker processes refresh requests in a dedicated goroutine,
// keeping CLOB API I/O off the main event loop.
func (mm *MarketMaker) refreshWorker() {
	for {
		select {
		case <-mm.done:
			return
		case req := <-mm.refreshCh:
			mm.refreshQuotes(req.market, req.fairYes)
		}
	}
}

// Stop signals the market maker to shut down.
func (mm *MarketMaker) Stop() {
	close(mm.done)
}

func (mm *MarketMaker) tick() {
	market := mm.discovery.GetCurrent()
	if market == nil || !market.AcceptingOrders {
		return
	}

	secsRemaining := time.Until(market.EndTime).Seconds()
	if secsRemaining <= 10 {
		// Don't quote in last 10 seconds — too risky
		return
	}

	mm.mu.Lock()

	// Detect new window — reset state
	if market.ConditionID != mm.currentWindowID {
		mm.currentWindowID = market.ConditionID
		mm.windowOpen = mm.chainlinkPrice
		mm.lastQuoteMid = 0
		mm.lastRefreshTime = time.Time{}
		mm.restingOrders = nil
	}

	chainlink := mm.chainlinkPrice
	windowOpen := mm.windowOpen
	mm.mu.Unlock()

	if chainlink <= 0 || windowOpen <= 0 {
		return
	}

	// Compute fair value
	fairYes := mm.computeFairValue(chainlink, windowOpen)

	// Check refresh triggers
	needRefresh := false

	mm.mu.Lock()
	if mm.lastQuoteMid == 0 {
		needRefresh = true // first quote
	} else if math.Abs(fairYes-mm.lastQuoteMid) > mmRefreshPriceMove {
		needRefresh = true // price moved
	} else if time.Since(mm.lastRefreshTime) > mmRefreshInterval {
		needRefresh = true // heartbeat
	}
	mm.mu.Unlock()

	if !needRefresh {
		return
	}

	// Send to async worker — non-blocking so tick() returns immediately
	select {
	case mm.refreshCh <- refreshRequest{market: market, fairYes: fairYes}:
	default:
		// Worker busy with previous refresh, skip this one
	}
}

func (mm *MarketMaker) computeFairValue(chainlink, windowOpen float64) float64 {
	// Primary: use directional model output when available and confident
	dirAction, dirConf := mm.directional.GetConfidence()
	if dirConf > 0.55 && (dirAction == "buy_yes" || dirAction == "buy_no") {
		if dirAction == "buy_yes" {
			return dirConf // calibrated P(up) from sidecar
		}
		return 1.0 - dirConf // P(up) = 1 - P(down)
	}

	// Fallback: linear formula from price move
	pctMove := (chainlink - windowOpen) / windowOpen
	fairYes := 0.50 + mmFallbackK*pctMove

	// Clamp to valid range
	if fairYes < 0.05 {
		fairYes = 0.05
	}
	if fairYes > 0.95 {
		fairYes = 0.95
	}

	return fairYes
}

func (mm *MarketMaker) refreshQuotes(market *execution.MarketInfo, fairYes float64) {
	// Cancel existing resting orders and wait for CLOB to release collateral
	mm.mu.Lock()
	oldOrders := mm.restingOrders
	mm.restingOrders = nil
	mm.mu.Unlock()

	if len(oldOrders) > 0 {
		for _, oid := range oldOrders {
			if err := mm.executor.CancelOrder(oid); err != nil {
				mm.log.Debug().Err(err).Str("order_id", oid).Msg("cancel failed")
			}
		}
		// Let CLOB release collateral before placing new orders
		time.Sleep(300 * time.Millisecond)
	}

	// Inventory check
	pos, hasPos := mm.position.GetPosition(market.ConditionID)
	var yesInv, noInv float64
	if hasPos {
		yesInv = pos.YesShares
		noInv = pos.NoShares
	}

	// Inventory skew
	skewYes := 0.0
	skewNo := 0.0
	netInv := yesInv - noInv
	if netInv > 0 {
		// Long YES — lower YES bid to discourage more, raise NO bid
		skewYes = -mmSkewCents
		skewNo = mmSkewCents
	} else if netInv < 0 {
		skewYes = mmSkewCents
		skewNo = -mmSkewCents
	}

	// Directional skew: when directional confidence > 0.65, skew in same direction
	dirAction, dirConf := mm.directional.GetConfidence()
	if dirConf > mmSkewThreshold {
		if dirAction == "buy_yes" {
			skewYes += mmSkewCents // widen ask, tighten bid on YES side
			skewNo -= mmSkewCents
		} else if dirAction == "buy_no" {
			skewYes -= mmSkewCents
			skewNo += mmSkewCents
		}
	}

	// Compute quote prices
	fairNo := 1.0 - fairYes
	yesBid := fairYes - mmHalfSpread + skewYes
	yesAsk := fairYes + mmHalfSpread + skewYes
	noBid := fairNo - mmHalfSpread + skewNo
	noAsk := fairNo + mmHalfSpread + skewNo

	// Clamp prices to [0.01, 0.99]
	yesBid = clampPrice(yesBid)
	yesAsk = clampPrice(yesAsk)
	noBid = clampPrice(noBid)
	noAsk = clampPrice(noAsk)

	// Available capital for MM (40% allocation), accounting for resting order collateral
	mmDeployed := mm.position.StrategyDeployed("market_maker")
	mmPending := mm.executor.PendingCapital("market_maker")
	mmAlloc := mm.bankroll * position.MarketMakerAlloc
	available := mmAlloc - mmDeployed - mmPending
	if available <= 0 {
		mm.log.Debug().Float64("deployed", mmDeployed).Msg("MM allocation exhausted")
		return
	}

	// Size per order — split available across 4 orders
	sizePerOrder := available / float64(mmMaxOrders)
	// Minimum $2.50 per order (ensures ≥5 shares at any price ≤0.50)
	if sizePerOrder < 2.50 {
		return
	}

	var orders []execution.OrderRequest

	// YES bid (buying YES at bid price) — skip if inventory maxed
	if yesInv < mmMaxInventory {
		shares := math.Max(sizePerOrder/yesBid, 5.0)
		orders = append(orders, execution.OrderRequest{
			StrategyID: "market_maker",
			TokenID:    market.UpTokenID,
			MarketID:   market.ConditionID,
			Side:       execution.Buy,
			Price:      yesBid,
			Size:       shares,
			OrderType:  execution.OrderTypePostOnly,
		})
	}

	// YES ask (selling YES = buying NO at complement)
	// In binary markets, selling YES at yesAsk is buying NO at 1-yesAsk
	// But we quote by placing a NO buy order
	if noInv < mmMaxInventory {
		shares := math.Max(sizePerOrder/noBid, 5.0)
		orders = append(orders, execution.OrderRequest{
			StrategyID: "market_maker",
			TokenID:    market.DownTokenID,
			MarketID:   market.ConditionID,
			Side:       execution.Buy,
			Price:      noBid,
			Size:       shares,
			OrderType:  execution.OrderTypePostOnly,
		})
	}

	// NO bid (buying NO at bid price)
	if noInv < mmMaxInventory && len(orders) < mmMaxOrders {
		shares := math.Max(sizePerOrder/noAsk, 5.0)
		if noAsk > 0.01 {
			orders = append(orders, execution.OrderRequest{
				StrategyID: "market_maker",
				TokenID:    market.DownTokenID,
				MarketID:   market.ConditionID,
				Side:       execution.Buy,
				Price:      noAsk,
				Size:       shares,
				OrderType:  execution.OrderTypePostOnly,
			})
		}
	}

	// YES ask complement
	if yesInv < mmMaxInventory && len(orders) < mmMaxOrders {
		shares := math.Max(sizePerOrder/yesAsk, 5.0)
		if yesAsk > 0.01 {
			orders = append(orders, execution.OrderRequest{
				StrategyID: "market_maker",
				TokenID:    market.UpTokenID,
				MarketID:   market.ConditionID,
				Side:       execution.Buy,
				Price:      yesAsk,
				Size:       shares,
				OrderType:  execution.OrderTypePostOnly,
			})
		}
	}

	if len(orders) == 0 {
		return
	}

	// Check allocation for each order
	if !mm.position.CanAllocate("market_maker", sizePerOrder) {
		mm.log.Debug().Msg("MM position allocation check failed")
		return
	}

	results, err := mm.executor.PlaceBatch(orders)
	if err != nil {
		mm.log.Warn().Err(err).Msg("MM batch order failed")
		return
	}

	mm.mu.Lock()
	mm.lastQuoteMid = fairYes
	mm.lastRefreshTime = time.Now()
	for _, r := range results {
		if r.Status == execution.StatusLive || r.Status == execution.StatusPaper {
			mm.restingOrders = append(mm.restingOrders, r.OrderID)
		}
	}
	mm.mu.Unlock()

	// Update metrics
	spreadBps := (yesAsk - yesBid) * 10000
	mm.metrics.MMSpreadBps.Set(spreadBps)
	mm.metrics.MMInventory.WithLabelValues("yes").Set(yesInv)
	mm.metrics.MMInventory.WithLabelValues("no").Set(noInv)

	mm.log.Info().
		Float64("fair_yes", fairYes).
		Float64("yes_bid", yesBid).
		Float64("yes_ask", yesAsk).
		Float64("no_bid", noBid).
		Float64("no_ask", noAsk).
		Float64("spread_bps", spreadBps).
		Int("orders", len(orders)).
		Float64("yes_inv", yesInv).
		Float64("no_inv", noInv).
		Msg("MM quotes refreshed")
}

func clampPrice(p float64) float64 {
	if p < 0.01 {
		return 0.01
	}
	if p > 0.99 {
		return 0.99
	}
	// Round to nearest cent
	return math.Round(p*100) / 100
}
