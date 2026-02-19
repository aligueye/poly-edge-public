package eventbus

import (
	"strconv"
	"sync"

	"github.com/rs/zerolog"
)

// EventType identifies which kind of event is being published.
type EventType string

const (
	EventBinanceBookUpdate    EventType = "binance.book"
	EventBinanceTrade         EventType = "binance.trade"
	EventBinanceLiquidation   EventType = "binance.liquidation"
	EventBinanceMarkPrice     EventType = "binance.mark_price"
	EventPolymarketBookUpdate EventType = "polymarket.book"
	EventPolymarketPrice      EventType = "polymarket.price"
	EventDeribitDVOL          EventType = "deribit.dvol"
)

// Event is the interface all bus events implement.
type Event interface {
	Type() EventType
}

// PriceLevel is a price/quantity pair used in order books.
type PriceLevel struct {
	Price float64
	Qty   float64
}

// ParsePriceLevel converts string price/qty to float64. Returns zero on error.
func ParsePriceLevel(price, qty string) PriceLevel {
	p, _ := strconv.ParseFloat(price, 64)
	q, _ := strconv.ParseFloat(qty, 64)
	return PriceLevel{Price: p, Qty: q}
}

// --- Concrete event types ---

type BinanceBookUpdate struct {
	Bids      []PriceLevel
	Asks      []PriceLevel
	Timestamp int64
}

func (e BinanceBookUpdate) Type() EventType { return EventBinanceBookUpdate }

type BinanceTrade struct {
	Price        float64
	Qty          float64
	IsBuyerMaker bool
	Timestamp    int64
}

func (e BinanceTrade) Type() EventType { return EventBinanceTrade }

type BinanceLiquidation struct {
	Side      string
	Price     float64
	Qty       float64
	Timestamp int64
}

func (e BinanceLiquidation) Type() EventType { return EventBinanceLiquidation }

type BinanceMarkPrice struct {
	MarkPrice   float64
	FundingRate float64
	Timestamp   int64
}

func (e BinanceMarkPrice) Type() EventType { return EventBinanceMarkPrice }

type PolymarketBookUpdate struct {
	MarketID   string
	YesBestBid float64
	YesBestAsk float64
	NoBestBid  float64
	NoBestAsk  float64
}

func (e PolymarketBookUpdate) Type() EventType { return EventPolymarketBookUpdate }

type PolymarketPrice struct {
	Price     float64
	Timestamp int64
}

func (e PolymarketPrice) Type() EventType { return EventPolymarketPrice }

type DeribitDVOL struct {
	Value     float64
	Timestamp int64
}

func (e DeribitDVOL) Type() EventType { return EventDeribitDVOL }

// --- Bus ---

const subscriberBuffer = 512

// Bus is a typed publish/subscribe event bus with fan-out.
type Bus struct {
	mu   sync.RWMutex
	subs map[EventType][]chan Event
	log  zerolog.Logger
}

// New creates a new event bus.
func New(logger zerolog.Logger) *Bus {
	return &Bus{
		subs: make(map[EventType][]chan Event),
		log:  logger.With().Str("component", "eventbus").Logger(),
	}
}

// Subscribe returns a buffered channel that receives events of the given type.
func (b *Bus) Subscribe(t EventType) <-chan Event {
	ch := make(chan Event, subscriberBuffer)
	b.mu.Lock()
	b.subs[t] = append(b.subs[t], ch)
	b.mu.Unlock()
	return ch
}

// Publish sends an event to all subscribers. Non-blocking: if a subscriber's
// channel is full, the event is dropped and a warning is logged.
func (b *Bus) Publish(e Event) {
	b.mu.RLock()
	subs := b.subs[e.Type()]
	b.mu.RUnlock()

	for _, ch := range subs {
		select {
		case ch <- e:
		default:
			// Drain oldest event to make room for latest
			select {
			case <-ch:
			default:
			}
			select {
			case ch <- e:
			default:
				b.log.Warn().
					Str("event", string(e.Type())).
					Msg("subscriber channel full, dropping event")
			}
		}
	}
}
