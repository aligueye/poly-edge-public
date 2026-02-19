package ws

import (
	"encoding/json"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/ali/poly-edge/internal/eventbus"
	"github.com/gorilla/websocket"
	"github.com/rs/zerolog"
)

const (
	clobWSURL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
	rtdsWSURL = "wss://ws-live-data.polymarket.com"

	// Default condition ID — overridden by market discovery at runtime.
	defaultConditionID = "0x0000000000000000000000000000000000000000000000000000000000000000"

	clobPingInterval = 10 * time.Second
)

// --- CLOB types ---

// ClobBookUpdate represents an order book snapshot from the CLOB WS.
type ClobBookUpdate struct {
	EventType string          `json:"event_type"`
	AssetID   string          `json:"asset_id"`
	Market    string          `json:"market"`
	Timestamp string          `json:"timestamp"`
	Hash      string          `json:"hash"`
	Buys      []ClobBookLevel `json:"buys"`
	Sells     []ClobBookLevel `json:"sells"`
}

// ClobBookLevel is a single price/size level.
type ClobBookLevel struct {
	Price string `json:"price"`
	Size  string `json:"size"`
}

// --- RTDS types ---

// RTDSMessage is the envelope for RTDS WebSocket messages.
type RTDSMessage struct {
	Topic     string          `json:"topic"`
	Type      string          `json:"type"`
	Timestamp int64           `json:"timestamp"`
	Payload   json.RawMessage `json:"payload"`
}

// RTDSCryptoPrice is the payload for a Chainlink crypto price update.
type RTDSCryptoPrice struct {
	Symbol    string  `json:"symbol"`
	Timestamp int64   `json:"timestamp"`
	Value     float64 `json:"value"`
}

// --- Client ---

// PolymarketClient manages both CLOB and RTDS WebSocket connections.
type PolymarketClient struct {
	clobConn *Conn
	rtdsConn *Conn
	log      zerolog.Logger
	bus      *eventbus.Bus

	conditionID string

	clobDone chan struct{}

	bookCount  atomic.Int64
	priceCount atomic.Int64
}

// SetEventBus sets the event bus for publishing parsed events.
func (p *PolymarketClient) SetEventBus(bus *eventbus.Bus) {
	p.bus = bus
}

// NewPolymarketClient creates a new Polymarket WS client with both connections.
func NewPolymarketClient(logger zerolog.Logger) *PolymarketClient {
	p := &PolymarketClient{
		log:         logger.With().Str("source", "polymarket").Logger(),
		conditionID: defaultConditionID,
		clobDone:    make(chan struct{}),
	}

	p.clobConn = NewConn(clobWSURL, nil, p.handleClobMessage, p.log.With().Str("feed", "clob").Logger())
	p.clobConn.SetOnConnect(p.onClobConnect)

	p.rtdsConn = NewConn(rtdsWSURL, nil, p.handleRtdsMessage, p.log.With().Str("feed", "rtds").Logger())
	p.rtdsConn.SetOnConnect(p.onRtdsConnect)

	return p
}

// IsConnected returns whether both CLOB and RTDS WebSockets are active.
func (p *PolymarketClient) IsConnected() bool {
	clob := p.clobConn != nil && p.clobConn.IsConnected()
	rtds := p.rtdsConn != nil && p.rtdsConn.IsConnected()
	return clob && rtds
}

// Connect starts both CLOB and RTDS WebSocket connections.
func (p *PolymarketClient) Connect() error {
	if err := p.clobConn.Connect(); err != nil {
		return err
	}
	go p.clobPingLoop()

	return p.rtdsConn.Connect()
}

// Disconnect stops both WebSocket connections.
func (p *PolymarketClient) Disconnect() {
	close(p.clobDone)
	p.clobConn.Disconnect()
	p.rtdsConn.Disconnect()
}

// --- CLOB ---

func (p *PolymarketClient) onClobConnect(c *Conn) error {
	sub := map[string]interface{}{
		"assets_ids": []string{p.conditionID},
		"type":       "market",
	}
	return c.WriteJSON(sub)
}

func (p *PolymarketClient) clobPingLoop() {
	ticker := time.NewTicker(clobPingInterval)
	defer ticker.Stop()
	for {
		select {
		case <-p.clobDone:
			return
		case <-ticker.C:
			if err := p.clobConn.WriteMessage(websocket.TextMessage, []byte("PING")); err != nil {
				p.log.Warn().Err(err).Msg("clob ping failed")
			}
		}
	}
}

func (p *PolymarketClient) handleClobMessage(_ int, data []byte) {
	if string(data) == "PONG" {
		return
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		p.log.Debug().Str("raw", string(data)).Msg("clob non-json message")
		return
	}

	eventType, _ := raw["event_type"].(string)
	switch eventType {
	case "book":
		var book ClobBookUpdate
		if err := json.Unmarshal(data, &book); err != nil {
			p.log.Error().Err(err).Msg("failed to parse clob book update")
			return
		}

		if p.bus != nil {
			var yesBid, yesAsk float64
			if len(book.Buys) > 0 {
				yesBid, _ = strconv.ParseFloat(book.Buys[0].Price, 64)
			}
			if len(book.Sells) > 0 {
				yesAsk, _ = strconv.ParseFloat(book.Sells[0].Price, 64)
			}
			p.bus.Publish(eventbus.PolymarketBookUpdate{
				MarketID:   book.Market,
				YesBestBid: yesBid,
				YesBestAsk: yesAsk,
				NoBestBid:  1.0 - yesAsk,
				NoBestAsk:  1.0 - yesBid,
			})
		}

		n := p.bookCount.Add(1)
		if n <= 5 {
			bestBid := ""
			bestAsk := ""
			if len(book.Buys) > 0 {
				bestBid = book.Buys[0].Price
			}
			if len(book.Sells) > 0 {
				bestAsk = book.Sells[0].Price
			}
			p.log.Debug().
				Str("asset", truncateID(book.AssetID)).
				Str("best_bid", bestBid).
				Str("best_ask", bestAsk).
				Int("buy_levels", len(book.Buys)).
				Int("sell_levels", len(book.Sells)).
				Msg("clob book update")
		}
	case "price_change", "last_trade_price", "tick_size_change":
		p.log.Debug().Str("event", eventType).Msg("clob event")
	default:
		p.log.Debug().Str("event_type", eventType).Msg("clob event")
	}
}

// --- RTDS ---

func (p *PolymarketClient) onRtdsConnect(c *Conn) error {
	sub := map[string]interface{}{
		"action": "subscribe",
		"subscriptions": []map[string]string{
			{
				"topic":   "crypto_prices_chainlink",
				"type":    "*",
				"filters": `{"symbol":"btc/usd"}`,
			},
		},
	}
	return c.WriteJSON(sub)
}

func (p *PolymarketClient) handleRtdsMessage(_ int, data []byte) {
	var msg RTDSMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		p.log.Debug().Str("raw", string(data)).Msg("rtds non-json message")
		return
	}

	if msg.Topic != "crypto_prices_chainlink" || msg.Type != "update" {
		return
	}

	var price RTDSCryptoPrice
	if err := json.Unmarshal(msg.Payload, &price); err != nil {
		p.log.Error().Err(err).Msg("failed to parse rtds price")
		return
	}

	if p.bus != nil {
		p.bus.Publish(eventbus.PolymarketPrice{
			Price: price.Value, Timestamp: price.Timestamp,
		})
	}

	n := p.priceCount.Add(1)
	if n <= 5 {
		p.log.Debug().
			Float64("btc_price", price.Value).
			Int64("ts", price.Timestamp).
			Msg("rtds btc/usd price")
	}
}

func truncateID(id string) string {
	if len(id) > 16 {
		return id[:16] + "..."
	}
	return id
}
