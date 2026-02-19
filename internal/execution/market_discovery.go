package execution

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/rs/zerolog"
)

const (
	gammaBaseURL     = "https://gamma-api.polymarket.com"
	discoveryPollInterval = 60 * time.Second
)

// MarketInfo describes an active BTC 5-minute binary market.
type MarketInfo struct {
	EventID      string
	EventSlug    string
	ConditionID  string
	UpTokenID    string   // clobTokenIds[0] = "Up" outcome
	DownTokenID  string   // clobTokenIds[1] = "Down" outcome
	UpPrice      float64
	DownPrice    float64
	StartTime    time.Time
	EndTime      time.Time
	Active       bool
	AcceptingOrders bool
	MinOrderSize float64
	TickSize     float64
}

// gammaEvent is the JSON shape from the Gamma API.
type gammaEvent struct {
	ID        string         `json:"id"`
	Slug      string         `json:"slug"`
	Active    bool           `json:"active"`
	Closed    bool           `json:"closed"`
	StartTime string         `json:"startTime"`
	Markets   []gammaMarket  `json:"markets"`
}

type gammaMarket struct {
	ID                      string `json:"id"`
	ConditionID             string `json:"conditionId"`
	Slug                    string `json:"slug"`
	Outcomes                string `json:"outcomes"`       // JSON string: '["Up","Down"]'
	OutcomePrices           string `json:"outcomePrices"`   // JSON string: '["0.515","0.485"]'
	ClobTokenIds            string `json:"clobTokenIds"`    // JSON string: '["123...","456..."]'
	Active                  bool   `json:"active"`
	AcceptingOrders         bool   `json:"acceptingOrders"`
	OrderMinSize            float64 `json:"orderMinSize"`
	OrderPriceMinTickSize   float64 `json:"orderPriceMinTickSize"`
}

// MarketDiscovery polls the Gamma API for active BTC 5-minute markets.
type MarketDiscovery struct {
	mu      sync.RWMutex
	current *MarketInfo
	next    *MarketInfo

	log    zerolog.Logger
	done   chan struct{}
	onChange func(MarketInfo) // called when a new market is discovered
}

// NewMarketDiscovery creates a market discovery service.
func NewMarketDiscovery(logger zerolog.Logger) *MarketDiscovery {
	return &MarketDiscovery{
		log:  logger.With().Str("component", "discovery").Logger(),
		done: make(chan struct{}),
	}
}

// OnChange registers a callback for when a new market is discovered.
func (d *MarketDiscovery) OnChange(fn func(MarketInfo)) {
	d.onChange = fn
}

// GetCurrent returns the currently active market (if any).
func (d *MarketDiscovery) GetCurrent() *MarketInfo {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.current == nil {
		return nil
	}
	copy := *d.current
	return &copy
}

// GetNext returns the upcoming market (if any).
func (d *MarketDiscovery) GetNext() *MarketInfo {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.next == nil {
		return nil
	}
	copy := *d.next
	return &copy
}

// Run starts polling the Gamma API. Blocks until Stop() is called.
func (d *MarketDiscovery) Run() {
	d.log.Info().Msg("market discovery started")

	// Initial poll
	d.poll()

	ticker := time.NewTicker(discoveryPollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-d.done:
			return
		case <-ticker.C:
			d.poll()
		}
	}
}

// Stop signals the discovery service to shut down.
func (d *MarketDiscovery) Stop() {
	close(d.done)
}

func (d *MarketDiscovery) poll() {
	now := time.Now()

	// Current 5-minute bucket
	currentBucket := now.Unix() - (now.Unix() % 300)
	currentMarket := d.fetchMarket(currentBucket)

	// Next 5-minute bucket
	nextBucket := currentBucket + 300
	nextMarket := d.fetchMarket(nextBucket)

	d.mu.Lock()
	oldCondition := ""
	if d.current != nil {
		oldCondition = d.current.ConditionID
	}

	d.current = currentMarket
	d.next = nextMarket
	d.mu.Unlock()

	if currentMarket != nil && currentMarket.ConditionID != oldCondition {
		d.log.Info().
			Str("event", currentMarket.EventSlug).
			Str("condition_id", truncateID(currentMarket.ConditionID)).
			Float64("up_price", currentMarket.UpPrice).
			Float64("down_price", currentMarket.DownPrice).
			Time("start", currentMarket.StartTime).
			Bool("accepting_orders", currentMarket.AcceptingOrders).
			Msg("new market discovered")

		if d.onChange != nil {
			d.onChange(*currentMarket)
		}
	}
}

func (d *MarketDiscovery) fetchMarket(bucket int64) *MarketInfo {
	slug := fmt.Sprintf("btc-updown-5m-%d", bucket)
	url := fmt.Sprintf("%s/events?slug=%s", gammaBaseURL, slug)

	resp, err := http.Get(url)
	if err != nil {
		d.log.Warn().Err(err).Str("slug", slug).Msg("gamma api request failed")
		return nil
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		d.log.Warn().Err(err).Msg("gamma api read failed")
		return nil
	}

	var events []gammaEvent
	if err := json.Unmarshal(body, &events); err != nil {
		d.log.Debug().Str("slug", slug).Msg("no market found for bucket")
		return nil
	}

	if len(events) == 0 || len(events[0].Markets) == 0 {
		return nil
	}

	event := events[0]
	market := event.Markets[0]

	// Parse clobTokenIds
	var tokenIDs []string
	if err := json.Unmarshal([]byte(market.ClobTokenIds), &tokenIDs); err != nil || len(tokenIDs) < 2 {
		d.log.Warn().Str("raw", market.ClobTokenIds).Msg("failed to parse clobTokenIds")
		return nil
	}

	// Parse outcomePrices
	var prices []string
	var upPrice, downPrice float64
	if err := json.Unmarshal([]byte(market.OutcomePrices), &prices); err == nil && len(prices) >= 2 {
		fmt.Sscanf(prices[0], "%f", &upPrice)
		fmt.Sscanf(prices[1], "%f", &downPrice)
	}

	// Parse startTime
	startTime, _ := time.Parse(time.RFC3339, event.StartTime)
	endTime := startTime.Add(5 * time.Minute)

	return &MarketInfo{
		EventID:         event.ID,
		EventSlug:       event.Slug,
		ConditionID:     market.ConditionID,
		UpTokenID:       tokenIDs[0],
		DownTokenID:     tokenIDs[1],
		UpPrice:         upPrice,
		DownPrice:       downPrice,
		StartTime:       startTime,
		EndTime:         endTime,
		Active:          market.Active,
		AcceptingOrders: market.AcceptingOrders,
		MinOrderSize:    market.OrderMinSize,
		TickSize:        market.OrderPriceMinTickSize,
	}
}

func truncateID(id string) string {
	if len(id) > 16 {
		return id[:16] + "..."
	}
	return id
}
