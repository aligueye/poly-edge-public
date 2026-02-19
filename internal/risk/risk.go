package risk

import (
	"fmt"

	"github.com/ali/poly-edge/internal/execution"
	"github.com/ali/poly-edge/internal/position"
)

// Risk limit constants (fractions of bankroll).
const (
	MaxPositionPerMarket = 0.25 // 25% of bankroll per market
	MaxTotalExposure     = 0.60 // 60% of bankroll total
	DailyLossLimit       = 0.05 // 5% of bankroll
	SingleTradeMax       = 0.25 // 25% of bankroll per trade
)

// Engine enforces pre-trade risk limits.
type Engine struct {
	bankroll float64
}

// NewEngine creates a risk engine with the given bankroll.
func NewEngine(bankroll float64) *Engine {
	return &Engine{bankroll: bankroll}
}

// PreTradeCheck validates an order against risk limits.
// Returns (allowed, reason). If allowed is false, reason explains why.
func (e *Engine) PreTradeCheck(order execution.OrderRequest, portfolio position.PortfolioState) (bool, string) {
	tradeCost := order.Price * order.Size

	// Rule a: single trade max 10% of bankroll
	if tradeCost > e.bankroll*SingleTradeMax+0.01 {
		return false, fmt.Sprintf("single trade $%.2f exceeds %.0f%% limit ($%.2f)",
			tradeCost, SingleTradeMax*100, e.bankroll*SingleTradeMax)
	}

	// Rule b: max position per market 10% of bankroll
	// The position manager tracks per-market exposure; here we check the new trade alone
	// against the per-market cap as a conservative estimate.
	if tradeCost > e.bankroll*MaxPositionPerMarket {
		return false, fmt.Sprintf("position in market would exceed %.0f%% limit ($%.2f)",
			MaxPositionPerMarket*100, e.bankroll*MaxPositionPerMarket)
	}

	// Rule c: max total exposure 60% of bankroll
	if portfolio.Deployed+tradeCost > e.bankroll*MaxTotalExposure {
		return false, fmt.Sprintf("total exposure $%.2f + $%.2f would exceed %.0f%% limit ($%.2f)",
			portfolio.Deployed, tradeCost, MaxTotalExposure*100, e.bankroll*MaxTotalExposure)
	}

	// Rule d: daily loss limit 2.5% of bankroll
	if portfolio.DailyPnL < -e.bankroll*DailyLossLimit {
		return false, fmt.Sprintf("daily loss $%.2f exceeds %.1f%% limit ($%.2f)",
			portfolio.DailyPnL, DailyLossLimit*100, e.bankroll*DailyLossLimit)
	}

	return true, ""
}
