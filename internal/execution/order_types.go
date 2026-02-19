package execution

import (
	"fmt"
	"time"
)

// Side represents the order side.
type Side int

const (
	Buy  Side = 0
	Sell Side = 1
)

func (s Side) String() string {
	switch s {
	case Buy:
		return "BUY"
	case Sell:
		return "SELL"
	default:
		return "UNKNOWN"
	}
}

// OrderType represents the order time-in-force type.
type OrderType string

const (
	OrderTypePostOnly OrderType = "POSTONLY" // GTC with postOnly=true — maker, zero fees
	OrderTypeGTC      OrderType = "GTC"     // Good-till-cancelled
	OrderTypeFOK      OrderType = "FOK"     // Fill-or-kill — urgent temporal arb
	OrderTypeFAK      OrderType = "FAK"     // Fill-and-kill
	OrderTypeGTD      OrderType = "GTD"     // Good-till-date
)

// OrderRequest is the input to the executor from a strategy.
type OrderRequest struct {
	StrategyID string    // Which strategy originated this order (required)
	TokenID    string    // ERC1155 conditional token ID (required)
	MarketID   string    // Condition ID / market (required)
	Side       Side      // Buy or Sell
	Price      float64   // Limit price, 0.01 - 0.99 for binary markets
	Size       float64   // Number of contracts (must be > 0)
	OrderType  OrderType // GTC, GTD, FOK, FAK, or POSTONLY
	Expiration int64     // Unix timestamp, required for GTD
}

// Validate checks that the OrderRequest fields are within acceptable bounds.
func (r *OrderRequest) Validate() error {
	if r.StrategyID == "" {
		return fmt.Errorf("strategy_id is required")
	}
	if r.TokenID == "" {
		return fmt.Errorf("token_id is required")
	}
	if r.MarketID == "" {
		return fmt.Errorf("market_id is required")
	}
	if r.Price < 0.01 || r.Price > 0.99 {
		return fmt.Errorf("price must be between 0.01 and 0.99, got %.4f", r.Price)
	}
	if r.Size <= 0 {
		return fmt.Errorf("size must be > 0, got %.4f", r.Size)
	}
	if r.OrderType == "" {
		return fmt.Errorf("order_type is required")
	}
	if r.OrderType == OrderTypeGTD && r.Expiration <= 0 {
		return fmt.Errorf("expiration is required for GTD orders")
	}
	return nil
}

// OrderStatus represents the lifecycle status of an order.
type OrderStatus string

const (
	StatusLive      OrderStatus = "live"
	StatusMatched   OrderStatus = "matched"
	StatusDelayed   OrderStatus = "delayed"
	StatusUnmatched OrderStatus = "unmatched"
	StatusCanceled  OrderStatus = "canceled"
	StatusFailed    OrderStatus = "failed"
	StatusPaper     OrderStatus = "paper"
)

// OrderResult is returned by the executor after placing an order.
type OrderResult struct {
	OrderID     string
	Status      OrderStatus
	ErrorMsg    string
	SubmittedAt time.Time
	LatencyMs   float64
	Request     OrderRequest
}

// FillEvent represents a fill notification.
type FillEvent struct {
	OrderID   string
	TradeID   string
	TokenID   string
	MarketID  string
	Side      Side
	Price     float64
	Size      float64
	Status    string // MATCHED, MINED, CONFIRMED, PAPER_FILL
	Timestamp time.Time
}
