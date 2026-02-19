package strategy

import (
	"github.com/ali/poly-edge/internal/eventbus"
	"github.com/ali/poly-edge/internal/execution"
	polygrpc "github.com/ali/poly-edge/internal/grpc"
	"github.com/ali/poly-edge/internal/metrics"
	"github.com/ali/poly-edge/internal/position"
	"github.com/rs/zerolog"
)

// Router dispatches both strategies concurrently.
// Directional is the primary strategy (60% capital allocation).
// MarketMaker is the complement (40% capital allocation).
type Router struct {
	directional *Directional
	marketMaker *MarketMaker
	posManager  *position.Manager
	log         zerolog.Logger
}

// NewRouter creates a strategy router wiring both strategies.
func NewRouter(
	signal *polygrpc.SignalClient,
	executor *execution.Executor,
	discovery *execution.MarketDiscovery,
	posManager *position.Manager,
	met *metrics.Metrics,
	bus *eventbus.Bus,
	bankroll float64,
	mmEnabled bool,
	logger zerolog.Logger,
) *Router {
	dir := NewDirectional(signal, executor, discovery, posManager, met, bankroll, logger)

	var mm *MarketMaker
	if mmEnabled {
		mm = NewMarketMaker(dir, executor, discovery, posManager, met, bus, bankroll, logger)
	}

	return &Router{
		directional: dir,
		marketMaker: mm,
		posManager:  posManager,
		log:         logger.With().Str("component", "router").Logger(),
	}
}

// Run starts strategies concurrently. Blocks until Stop() is called.
func (r *Router) Run() {
	if r.marketMaker != nil {
		r.log.Info().Msg("strategy router starting — directional (60%) + market maker (40%)")
		go r.directional.Run()
		r.marketMaker.Run() // blocks
	} else {
		r.log.Info().Msg("strategy router starting — directional only (100%)")
		r.directional.Run() // blocks
	}
}

// Stop shuts down strategies.
func (r *Router) Stop() {
	r.log.Info().Msg("strategy router stopping")
	r.directional.Stop()
	if r.marketMaker != nil {
		r.marketMaker.Stop()
	}
}

// DirectionalStats returns the directional strategy's session stats.
func (r *Router) DirectionalStats() (total, correct int, avgConf float64) {
	return r.directional.Stats()
}

// RecordResolution handles a market resolution: frees capital in position manager
// and notifies the directional strategy for accuracy tracking.
func (r *Router) RecordResolution(conditionID, winnerSide string) {
	// Determine if the directional bet won
	betSide := r.directional.GetBetSide(conditionID)
	if betSide != "" {
		won := betSide == winnerSide
		r.directional.RecordResolution(conditionID, won)
	}
}
