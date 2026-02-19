package position

import (
	"sync"
	"time"

	"github.com/rs/zerolog"
)

// Strategy capital allocation percentages.
const (
	DirectionalAlloc = 0.60
	MarketMakerAlloc = 0.40
)

// MarketPosition tracks shares held in a single market.
type MarketPosition struct {
	MarketID    string
	ConditionID string
	YesTokenID  string
	NoTokenID   string
	YesShares   float64
	NoShares    float64
	AvgCostYes  float64
	AvgCostNo   float64
	StrategyID  string
}

// CostBasis returns the total capital deployed in this position.
func (p *MarketPosition) CostBasis() float64 {
	return p.YesShares*p.AvgCostYes + p.NoShares*p.AvgCostNo
}

// PortfolioState is a snapshot of the overall portfolio.
type PortfolioState struct {
	TotalBankroll   float64
	Deployed        float64
	Available       float64
	UnrealizedPnL   float64
	RealizedPnL     float64
	DailyPnL        float64
	DailyPnLResetAt time.Time
}

// Manager tracks open positions and P&L across strategies.
type Manager struct {
	mu        sync.RWMutex
	positions map[string]*MarketPosition // key: marketID
	bankroll  float64
	deployed  float64

	realizedPnL float64
	dailyPnL    float64
	dailyReset  time.Time

	log zerolog.Logger
}

// NewManager creates a new position manager with the given bankroll.
func NewManager(bankroll float64, logger zerolog.Logger) *Manager {
	now := time.Now().UTC()
	return &Manager{
		positions:  make(map[string]*MarketPosition),
		bankroll:   bankroll,
		dailyReset: time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC),
		log:        logger.With().Str("component", "position").Logger(),
	}
}

// RecordFill records a fill event, updating position and deployed capital.
func (m *Manager) RecordFill(marketID, side string, shares, price float64, strategyID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.checkDailyReset()

	pos, exists := m.positions[marketID]
	if !exists {
		pos = &MarketPosition{MarketID: marketID, StrategyID: strategyID}
		m.positions[marketID] = pos
	}

	cost := shares * price
	switch side {
	case "YES":
		totalShares := pos.YesShares + shares
		if totalShares > 0 {
			pos.AvgCostYes = (pos.YesShares*pos.AvgCostYes + cost) / totalShares
		}
		pos.YesShares = totalShares
	case "NO":
		totalShares := pos.NoShares + shares
		if totalShares > 0 {
			pos.AvgCostNo = (pos.NoShares*pos.AvgCostNo + cost) / totalShares
		}
		pos.NoShares = totalShares
	}

	m.deployed += cost

	m.log.Info().
		Str("market", marketID).
		Str("side", side).
		Float64("shares", shares).
		Float64("price", price).
		Str("strategy", strategyID).
		Float64("deployed", m.deployed).
		Msg("fill recorded")
}

// RecordResolution handles market resolution, computing realized PnL.
// winnerSide is "YES" or "NO". Winners get $1.00 per share, losers get $0.00.
func (m *Manager) RecordResolution(marketID, winnerSide string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.checkDailyReset()

	pos, exists := m.positions[marketID]
	if !exists {
		return
	}

	var pnl float64
	switch winnerSide {
	case "YES":
		pnl = pos.YesShares*(1.0-pos.AvgCostYes) - pos.NoShares*pos.AvgCostNo
	case "NO":
		pnl = pos.NoShares*(1.0-pos.AvgCostNo) - pos.YesShares*pos.AvgCostYes
	}

	m.deployed -= pos.CostBasis()
	if m.deployed < 0 {
		m.deployed = 0
	}

	m.realizedPnL += pnl
	m.dailyPnL += pnl

	m.log.Info().
		Str("market", marketID).
		Str("winner", winnerSide).
		Float64("pnl", pnl).
		Float64("realized_total", m.realizedPnL).
		Float64("daily_pnl", m.dailyPnL).
		Msg("resolution recorded")

	delete(m.positions, marketID)
}

// GetPosition returns the position for a market (copy).
func (m *Manager) GetPosition(marketID string) (MarketPosition, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	pos, ok := m.positions[marketID]
	if !ok {
		return MarketPosition{}, false
	}
	return *pos, true
}

// GetPortfolio returns the current portfolio state.
func (m *Manager) GetPortfolio() PortfolioState {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return PortfolioState{
		TotalBankroll:   m.bankroll,
		Deployed:        m.deployed,
		Available:       m.bankroll - m.deployed + m.realizedPnL,
		RealizedPnL:     m.realizedPnL,
		DailyPnL:        m.dailyPnL,
		DailyPnLResetAt: m.dailyReset,
	}
}

// CanAllocate checks if a strategy has enough remaining allocation.
func (m *Manager) CanAllocate(strategyID string, amount float64) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	allocPct := allocationForStrategy(strategyID)
	maxAlloc := m.bankroll * allocPct

	var used float64
	for _, pos := range m.positions {
		if pos.StrategyID == strategyID {
			used += pos.CostBasis()
		}
	}

	return used+amount <= maxAlloc
}

// StrategyDeployed returns how much capital a specific strategy has deployed.
func (m *Manager) StrategyDeployed(strategyID string) float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var used float64
	for _, pos := range m.positions {
		if pos.StrategyID == strategyID {
			used += pos.CostBasis()
		}
	}
	return used
}

func allocationForStrategy(strategyID string) float64 {
	switch strategyID {
	case "directional":
		return DirectionalAlloc
	case "market_maker":
		return MarketMakerAlloc
	default:
		return 0
	}
}

func (m *Manager) checkDailyReset() {
	now := time.Now().UTC()
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, time.UTC)
	if today.After(m.dailyReset) {
		m.dailyPnL = 0
		m.dailyReset = today
	}
}
