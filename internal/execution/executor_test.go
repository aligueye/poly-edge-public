package execution

import (
	"testing"

	"github.com/ali/poly-edge/internal/metrics"
	"github.com/ali/poly-edge/internal/position"
	"github.com/rs/zerolog"
)

// testRisk is a simple RiskChecker that always allows (or always rejects).
type testRisk struct {
	allow  bool
	reason string
}

func (r *testRisk) PreTradeCheck(_ OrderRequest, _ position.PortfolioState) (bool, string) {
	return r.allow, r.reason
}

func newTestExecutor(allow bool) (*Executor, *position.Manager) {
	pos := position.NewManager(1000, zerolog.Nop())
	met := metrics.NewNoop()
	risk := &testRisk{allow: allow, reason: "test rejection"}
	exec := NewExecutor(risk, pos, met, nil, true, zerolog.Nop())
	return exec, pos
}

func testOrder() OrderRequest {
	return OrderRequest{
		StrategyID: "directional",
		TokenID:    "tok1",
		MarketID:   "mkt1",
		Side:       Buy,
		Price:      0.52,
		Size:       10,
		OrderType:  OrderTypePostOnly,
	}
}

func TestPaperFill_RecordsPosition(t *testing.T) {
	exec, pos := newTestExecutor(true)

	result, err := exec.PlaceOrder(testOrder())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Status != StatusPaper {
		t.Errorf("expected status paper, got %s", result.Status)
	}
	if result.OrderID == "" {
		t.Error("expected non-empty order ID")
	}

	// Position should be updated
	p, ok := pos.GetPosition("mkt1")
	if !ok {
		t.Fatal("expected position to exist")
	}
	if p.YesShares != 10 {
		t.Errorf("expected 10 YES shares, got %f", p.YesShares)
	}
	if p.AvgCostYes != 0.52 {
		t.Errorf("expected avg cost 0.52, got %f", p.AvgCostYes)
	}
}

func TestPaperFill_RiskRejection(t *testing.T) {
	exec, _ := newTestExecutor(false)

	result, err := exec.PlaceOrder(testOrder())
	if err == nil {
		t.Fatal("expected error from risk rejection")
	}
	if result.Status != StatusFailed {
		t.Errorf("expected status failed, got %s", result.Status)
	}
}

func TestPaperFill_ValidationError(t *testing.T) {
	exec, _ := newTestExecutor(true)

	bad := testOrder()
	bad.TokenID = "" // required field

	result, err := exec.PlaceOrder(bad)
	if err == nil {
		t.Fatal("expected validation error")
	}
	if result.Status != StatusFailed {
		t.Errorf("expected status failed, got %s", result.Status)
	}
}

func TestPaperCancel(t *testing.T) {
	exec, _ := newTestExecutor(true)

	// Cancel should succeed even with no matching order (paper mode)
	if err := exec.CancelOrder("nonexistent"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestPaperCancelAllForMarket(t *testing.T) {
	exec, _ := newTestExecutor(true)

	if err := exec.CancelAllForMarket("mkt1"); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestBatchMaxSize(t *testing.T) {
	exec, _ := newTestExecutor(true)

	orders := make([]OrderRequest, 16)
	_, err := exec.PlaceBatch(orders)
	if err == nil {
		t.Error("expected error for batch > 15")
	}
}

func TestBatchPlacesOrders(t *testing.T) {
	exec, pos := newTestExecutor(true)

	orders := []OrderRequest{
		{StrategyID: "directional", TokenID: "tok1", MarketID: "mkt1", Side: Buy, Price: 0.50, Size: 5, OrderType: OrderTypePostOnly},
		{StrategyID: "directional", TokenID: "tok2", MarketID: "mkt2", Side: Buy, Price: 0.55, Size: 10, OrderType: OrderTypePostOnly},
	}

	results, err := exec.PlaceBatch(orders)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	for i, r := range results {
		if r.Status != StatusPaper {
			t.Errorf("order %d: expected paper status, got %s", i, r.Status)
		}
	}

	// Both positions should exist
	if _, ok := pos.GetPosition("mkt1"); !ok {
		t.Error("expected position mkt1")
	}
	if _, ok := pos.GetPosition("mkt2"); !ok {
		t.Error("expected position mkt2")
	}
}
