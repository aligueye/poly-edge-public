package risk

import (
	"testing"

	"github.com/ali/poly-edge/internal/execution"
	"github.com/ali/poly-edge/internal/position"
)

func portfolio(deployed, dailyPnL float64) position.PortfolioState {
	return position.PortfolioState{
		TotalBankroll: 1000,
		Deployed:      deployed,
		Available:     1000 - deployed,
		DailyPnL:      dailyPnL,
	}
}

func order(price, size float64) execution.OrderRequest {
	return execution.OrderRequest{
		StrategyID: "directional",
		TokenID:    "tok1",
		MarketID:   "mkt1",
		Side:       execution.Buy,
		Price:      price,
		Size:       size,
		OrderType:  execution.OrderTypePostOnly,
	}
}

func TestSingleTradeMax(t *testing.T) {
	e := NewEngine(1000)

	// 25% of $1000 = $250. Trade of $240 should pass.
	ok, _ := e.PreTradeCheck(order(0.50, 480), portfolio(0, 0)) // 0.50 * 480 = $240
	if !ok {
		t.Error("$240 trade should be allowed (limit $250)")
	}

	// Trade of $260 should fail.
	ok, reason := e.PreTradeCheck(order(0.50, 520), portfolio(0, 0)) // 0.50 * 520 = $260
	if ok {
		t.Error("$260 trade should be rejected (limit $250)")
	}
	if reason == "" {
		t.Error("expected rejection reason")
	}
}

func TestMaxPositionPerMarket(t *testing.T) {
	// SingleTradeMax and MaxPositionPerMarket are both 25%.
	// The per-market rule becomes meaningful with cumulative tracking.
	// For now, verify that a trade exceeding 25% of bankroll is rejected.
	e := NewEngine(1_000_000)

	// $240,000 trade: passes both limits ($250k each).
	ok, _ := e.PreTradeCheck(order(0.50, 480_000), portfolio(0, 0)) // 0.50 * 480,000 = $240,000
	if !ok {
		t.Error("$240,000 trade should be allowed")
	}

	// $260,000 trade: exceeds both limits, rejected.
	ok, _ = e.PreTradeCheck(order(0.50, 520_000), portfolio(0, 0)) // 0.50 * 520,000 = $260,000
	if ok {
		t.Error("$260,000 trade should be rejected")
	}
}

func TestMaxTotalExposure(t *testing.T) {
	e := NewEngine(1000)

	// 60% of $1000 = $600. Already deployed $590, new trade $15 = $605 → reject.
	ok, _ := e.PreTradeCheck(order(0.50, 30), portfolio(590, 0)) // 0.50 * 30 = $15
	if ok {
		t.Error("should reject: $590 + $15 = $605 exceeds $600 limit")
	}

	// $590 deployed, new trade $10 = $600 → allow (at the limit).
	ok, _ = e.PreTradeCheck(order(0.50, 20), portfolio(590, 0)) // 0.50 * 20 = $10
	if !ok {
		t.Error("should allow: $590 + $10 = $600 exactly at limit")
	}
}

func TestDailyLossLimit(t *testing.T) {
	e := NewEngine(1000)

	// 5% of $1000 = $50. Daily PnL of -$51 should block all new trades.
	ok, _ := e.PreTradeCheck(order(0.50, 2), portfolio(0, -51))
	if ok {
		t.Error("should reject: daily loss -$51 exceeds -$50 limit")
	}

	// Daily PnL of -$40 should allow trades.
	ok, _ = e.PreTradeCheck(order(0.50, 2), portfolio(0, -40))
	if !ok {
		t.Error("should allow: daily loss -$40 within -$50 limit")
	}
}

func TestAllRulesPass(t *testing.T) {
	e := NewEngine(1000)

	ok, reason := e.PreTradeCheck(order(0.50, 10), portfolio(100, -5)) // $5 trade, $105 total, -$5 daily
	if !ok {
		t.Errorf("all rules should pass, got rejection: %s", reason)
	}
}
