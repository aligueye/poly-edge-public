package ws

import (
	"encoding/json"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/ali/poly-edge/internal/eventbus"
	"github.com/rs/zerolog"
)

const binanceURL = "wss://fstream.binance.com/stream?streams=btcusdt@depth20@100ms/btcusdt@aggTrade/btcusdt@forceOrder"

// DepthUpdate represents a Binance partial book depth snapshot.
type DepthUpdate struct {
	EventType string     `json:"e"`
	EventTime int64      `json:"E"`
	Bids      [][]string `json:"b"`
	Asks      [][]string `json:"a"`
}

// AggTrade represents a Binance aggregate trade.
type AggTrade struct {
	Price        string `json:"p"`
	Quantity     string `json:"q"`
	IsBuyerMaker bool   `json:"m"`
	TradeTime    int64  `json:"T"`
}

// ForceOrder represents a Binance liquidation event.
type ForceOrder struct {
	Order struct {
		Side      string `json:"S"`
		Price     string `json:"p"`
		Quantity  string `json:"q"`
		TradeTime int64  `json:"T"`
	} `json:"o"`
}

// OrderBookLevel is a single price/quantity level.
type OrderBookLevel struct {
	Price    string
	Quantity string
}

// OrderBook holds the local 20-level order book snapshot.
type OrderBook struct {
	mu   sync.RWMutex
	Bids [20]OrderBookLevel
	Asks [20]OrderBookLevel
}

// BinanceClient manages the Binance futures WebSocket.
type BinanceClient struct {
	conn *Conn
	log  zerolog.Logger
	bus  *eventbus.Bus
	Book OrderBook

	depthCount      atomic.Int64
	aggTradeCount   atomic.Int64
	forceOrderCount atomic.Int64
}

// SetEventBus sets the event bus for publishing parsed events.
func (b *BinanceClient) SetEventBus(bus *eventbus.Bus) {
	b.bus = bus
}

// NewBinanceClient creates a new Binance WS client.
func NewBinanceClient(logger zerolog.Logger) *BinanceClient {
	b := &BinanceClient{
		log: logger.With().Str("source", "binance").Logger(),
	}
	b.conn = NewConn(binanceURL, nil, b.handleMessage, b.log)
	return b
}

// Connect starts the Binance WebSocket connection.
func (b *BinanceClient) Connect() error {
	return b.conn.Connect()
}

// IsConnected returns whether the WebSocket is currently active.
func (b *BinanceClient) IsConnected() bool {
	return b.conn != nil && b.conn.IsConnected()
}

// Disconnect stops the Binance WebSocket connection.
func (b *BinanceClient) Disconnect() {
	b.conn.Disconnect()
}

func (b *BinanceClient) handleMessage(_ int, data []byte) {
	var wrapper struct {
		Stream string          `json:"stream"`
		Data   json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(data, &wrapper); err != nil {
		b.log.Error().Err(err).Msg("failed to parse binance wrapper")
		return
	}

	switch wrapper.Stream {
	case "btcusdt@depth20@100ms":
		b.handleDepth(wrapper.Data)
	case "btcusdt@aggTrade":
		b.handleAggTrade(wrapper.Data)
	case "btcusdt@forceOrder":
		b.handleForceOrder(wrapper.Data)
	default:
		b.log.Warn().Str("stream", wrapper.Stream).Msg("unknown stream")
	}
}

func (b *BinanceClient) handleDepth(data json.RawMessage) {
	var d DepthUpdate
	if err := json.Unmarshal(data, &d); err != nil {
		b.log.Error().Err(err).Msg("failed to parse depth update")
		return
	}

	b.Book.mu.Lock()
	for i := 0; i < len(d.Bids) && i < 20; i++ {
		b.Book.Bids[i] = OrderBookLevel{Price: d.Bids[i][0], Quantity: d.Bids[i][1]}
	}
	for i := 0; i < len(d.Asks) && i < 20; i++ {
		b.Book.Asks[i] = OrderBookLevel{Price: d.Asks[i][0], Quantity: d.Asks[i][1]}
	}
	b.Book.mu.Unlock()

	if b.bus != nil {
		bids := make([]eventbus.PriceLevel, len(d.Bids))
		asks := make([]eventbus.PriceLevel, len(d.Asks))
		for i, bid := range d.Bids {
			bids[i] = eventbus.ParsePriceLevel(bid[0], bid[1])
		}
		for i, ask := range d.Asks {
			asks[i] = eventbus.ParsePriceLevel(ask[0], ask[1])
		}
		b.bus.Publish(eventbus.BinanceBookUpdate{
			Bids: bids, Asks: asks, Timestamp: d.EventTime,
		})
	}

	n := b.depthCount.Add(1)
	if n <= 5 {
		b.log.Debug().
			Str("best_bid", d.Bids[0][0]).
			Str("best_ask", d.Asks[0][0]).
			Int64("event_time", d.EventTime).
			Msg("depth update")
	}
}

func (b *BinanceClient) handleAggTrade(data json.RawMessage) {
	var t AggTrade
	if err := json.Unmarshal(data, &t); err != nil {
		b.log.Error().Err(err).Msg("failed to parse agg trade")
		return
	}

	if b.bus != nil {
		price, _ := strconv.ParseFloat(t.Price, 64)
		qty, _ := strconv.ParseFloat(t.Quantity, 64)
		b.bus.Publish(eventbus.BinanceTrade{
			Price: price, Qty: qty, IsBuyerMaker: t.IsBuyerMaker, Timestamp: t.TradeTime,
		})
	}

	n := b.aggTradeCount.Add(1)
	if n <= 5 {
		b.log.Debug().
			Str("price", t.Price).
			Str("qty", t.Quantity).
			Bool("buyer_maker", t.IsBuyerMaker).
			Int64("trade_time", t.TradeTime).
			Msg("agg trade")
	}
}

func (b *BinanceClient) handleForceOrder(data json.RawMessage) {
	var f ForceOrder
	if err := json.Unmarshal(data, &f); err != nil {
		b.log.Error().Err(err).Msg("failed to parse force order")
		return
	}

	if b.bus != nil {
		price, _ := strconv.ParseFloat(f.Order.Price, 64)
		qty, _ := strconv.ParseFloat(f.Order.Quantity, 64)
		b.bus.Publish(eventbus.BinanceLiquidation{
			Side: f.Order.Side, Price: price, Qty: qty, Timestamp: f.Order.TradeTime,
		})
	}

	n := b.forceOrderCount.Add(1)
	if n <= 5 {
		b.log.Debug().
			Str("side", f.Order.Side).
			Str("price", f.Order.Price).
			Str("qty", f.Order.Quantity).
			Int64("trade_time", f.Order.TradeTime).
			Msg("force order (liquidation)")
	}
}
