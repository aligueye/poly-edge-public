package position

import (
	"testing"

	"github.com/rs/zerolog"
)

func newTestManager(bankroll float64) *Manager {
	return NewManager(bankroll, zerolog.Nop())
}

func TestRecordFill_SingleYes(t *testing.T) {
	m := newTestManager(1000)
	m.RecordFill("mkt1", "YES", 10, 0.52, "directional")

	pos, ok := m.GetPosition("mkt1")
	if !ok {
		t.Fatal("expected position to exist")
	}
	if pos.YesShares != 10 {
		t.Errorf("expected 10 YES shares, got %f", pos.YesShares)
	}
	if pos.AvgCostYes != 0.52 {
		t.Errorf("expected avg cost 0.52, got %f", pos.AvgCostYes)
	}
	if pos.NoShares != 0 {
		t.Errorf("expected 0 NO shares, got %f", pos.NoShares)
	}
}

func TestRecordFill_WeightedAverage(t *testing.T) {
	m := newTestManager(1000)
	m.RecordFill("mkt1", "YES", 10, 0.50, "directional")
	m.RecordFill("mkt1", "YES", 10, 0.60, "directional")

	pos, _ := m.GetPosition("mkt1")
	if pos.YesShares != 20 {
		t.Errorf("expected 20 shares, got %f", pos.YesShares)
	}
	// Weighted avg: (10*0.50 + 10*0.60) / 20 = 0.55
	if abs(pos.AvgCostYes-0.55) > 0.001 {
		t.Errorf("expected avg cost ~0.55, got %f", pos.AvgCostYes)
	}
}

func TestRecordResolution_YesWins(t *testing.T) {
	m := newTestManager(1000)
	m.RecordFill("mkt1", "YES", 10, 0.52, "directional")
	m.RecordResolution("mkt1", "YES")

	// PnL: 10 * (1.00 - 0.52) = 4.80
	portfolio := m.GetPortfolio()
	if abs(portfolio.RealizedPnL-4.80) > 0.01 {
		t.Errorf("expected realized PnL ~4.80, got %f", portfolio.RealizedPnL)
	}
	if abs(portfolio.DailyPnL-4.80) > 0.01 {
		t.Errorf("expected daily PnL ~4.80, got %f", portfolio.DailyPnL)
	}
	// Position should be removed
	if _, ok := m.GetPosition("mkt1"); ok {
		t.Error("position should be removed after resolution")
	}
}

func TestRecordResolution_NoWins(t *testing.T) {
	m := newTestManager(1000)
	m.RecordFill("mkt1", "YES", 10, 0.52, "directional")
	m.RecordResolution("mkt1", "NO")

	// PnL: -(10 * 0.52) = -5.20
	portfolio := m.GetPortfolio()
	if abs(portfolio.RealizedPnL-(-5.20)) > 0.01 {
		t.Errorf("expected realized PnL ~-5.20, got %f", portfolio.RealizedPnL)
	}
}

func TestRecordResolution_BothSides(t *testing.T) {
	m := newTestManager(1000)
	m.RecordFill("mkt1", "YES", 10, 0.52, "directional")
	m.RecordFill("mkt1", "NO", 5, 0.48, "directional")
	m.RecordResolution("mkt1", "YES")

	// YES wins: 10*(1.0-0.52) - 5*0.48 = 4.80 - 2.40 = 2.40
	portfolio := m.GetPortfolio()
	if abs(portfolio.RealizedPnL-2.40) > 0.01 {
		t.Errorf("expected realized PnL ~2.40, got %f", portfolio.RealizedPnL)
	}
}

func TestCanAllocate_Directional(t *testing.T) {
	m := newTestManager(1000)

	// Directional gets 60% = $600
	if !m.CanAllocate("directional", 500) {
		t.Error("should be able to allocate $500 of $600 directional budget")
	}
	if m.CanAllocate("directional", 700) {
		t.Error("should not be able to allocate $700 of $600 directional budget")
	}
}

func TestCanAllocate_MarketMaker(t *testing.T) {
	m := newTestManager(1000)

	// MM gets 40% = $400
	if !m.CanAllocate("market_maker", 300) {
		t.Error("should be able to allocate $300 of $400 MM budget")
	}
	if m.CanAllocate("market_maker", 500) {
		t.Error("should not be able to allocate $500 of $400 MM budget")
	}
}

func TestCanAllocate_AfterFill(t *testing.T) {
	m := newTestManager(1000)
	// Use $500 of directional budget (10 shares * $0.50 = $5.00, not $500)
	// Let's use larger amounts: 500 shares at $0.50 = $250
	m.RecordFill("mkt1", "YES", 500, 0.50, "directional")

	// $250 used of $600 budget, $350 remaining
	if !m.CanAllocate("directional", 300) {
		t.Error("should be able to allocate $300 with $350 remaining")
	}
	if m.CanAllocate("directional", 400) {
		t.Error("should not be able to allocate $400 with $350 remaining")
	}
}

func TestGetPortfolio_Available(t *testing.T) {
	m := newTestManager(1000)
	m.RecordFill("mkt1", "YES", 100, 0.50, "directional")

	portfolio := m.GetPortfolio()
	// deployed = 100 * 0.50 = 50
	if abs(portfolio.Deployed-50) > 0.01 {
		t.Errorf("expected deployed 50, got %f", portfolio.Deployed)
	}
	// available = 1000 - 50 + 0 = 950
	if abs(portfolio.Available-950) > 0.01 {
		t.Errorf("expected available 950, got %f", portfolio.Available)
	}
}

func TestDeployedFreedOnResolution(t *testing.T) {
	m := newTestManager(1000)
	m.RecordFill("mkt1", "YES", 100, 0.50, "directional")
	m.RecordResolution("mkt1", "YES")

	portfolio := m.GetPortfolio()
	if portfolio.Deployed != 0 {
		t.Errorf("expected deployed 0 after resolution, got %f", portfolio.Deployed)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
