package execution

import (
	"sync"
	"time"

	"github.com/ali/poly-edge/internal/position"
	"github.com/rs/zerolog"
)

const (
	resolverPollInterval = 1 * time.Second
	resolverBuffer       = 15 * time.Second // wait after endTime before resolving
)

type windowInfo struct {
	conditionID string
	endTime     time.Time
	openPrice   float64
}

// Resolver tracks market windows and determines the winner (YES/NO)
// by comparing BTC price at window close vs open.
type Resolver struct {
	posManager *position.Manager
	getPrice   func() float64 // returns current BTC/USD price
	log        zerolog.Logger

	mu            sync.Mutex
	activeWindows map[string]*windowInfo // conditionID → window
	onResolution  func(conditionID, winnerSide string)

	done chan struct{}
}

// NewResolver creates a market resolution tracker.
func NewResolver(
	posManager *position.Manager,
	getPrice func() float64,
	logger zerolog.Logger,
) *Resolver {
	return &Resolver{
		posManager:    posManager,
		getPrice:      getPrice,
		log:           logger.With().Str("component", "resolver").Logger(),
		activeWindows: make(map[string]*windowInfo),
		done:          make(chan struct{}),
	}
}

// TrackMarket registers a market window for resolution tracking.
func (r *Resolver) TrackMarket(conditionID string, endTime time.Time, openPrice float64) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.activeWindows[conditionID]; exists {
		return // already tracking
	}

	r.activeWindows[conditionID] = &windowInfo{
		conditionID: conditionID,
		endTime:     endTime,
		openPrice:   openPrice,
	}

	r.log.Info().
		Str("condition_id", conditionID).
		Time("end_time", endTime).
		Float64("open_price", openPrice).
		Msg("tracking market for resolution")
}

// OnResolution registers a callback invoked when a market resolves.
func (r *Resolver) OnResolution(fn func(conditionID, winnerSide string)) {
	r.onResolution = fn
}

// Run starts the resolution polling loop. Blocks until Stop().
func (r *Resolver) Run() {
	r.log.Info().Msg("resolver started")

	ticker := time.NewTicker(resolverPollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-r.done:
			return
		case <-ticker.C:
			r.checkResolutions()
		}
	}
}

// Stop shuts down the resolver.
func (r *Resolver) Stop() {
	close(r.done)
}

func (r *Resolver) checkResolutions() {
	now := time.Now()
	closePrice := r.getPrice()
	if closePrice <= 0 {
		return
	}

	r.mu.Lock()
	var resolved []windowInfo
	for cid, w := range r.activeWindows {
		if now.After(w.endTime.Add(resolverBuffer)) {
			resolved = append(resolved, *w)
			delete(r.activeWindows, cid)
		}
	}
	r.mu.Unlock()

	for _, w := range resolved {
		winnerSide := "NO"
		if closePrice > w.openPrice {
			winnerSide = "YES"
		}

		r.posManager.RecordResolution(w.conditionID, winnerSide)

		r.log.Info().
			Str("condition_id", w.conditionID).
			Str("winner", winnerSide).
			Float64("open_price", w.openPrice).
			Float64("close_price", closePrice).
			Msg("market resolved")

		if r.onResolution != nil {
			r.onResolution(w.conditionID, winnerSide)
		}
	}
}
