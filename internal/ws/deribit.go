package ws

import (
	"encoding/json"

	"github.com/ali/poly-edge/internal/eventbus"
	"github.com/rs/zerolog"
)

const deribitURL = "wss://www.deribit.com/ws/api/v2"

// DeribitRPCRequest is a JSON-RPC 2.0 request.
type DeribitRPCRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      int         `json:"id"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params"`
}

// DeribitRPCResponse is a JSON-RPC 2.0 response envelope.
type DeribitRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int            `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
}

// DeribitSubscriptionData holds the subscription notification data.
type DeribitSubscriptionData struct {
	Channel string          `json:"channel"`
	Data    json.RawMessage `json:"data"`
}

// DeribitDVOL represents the DVOL volatility index data.
type DeribitDVOL struct {
	Volatility float64 `json:"volatility"`
	IndexName  string  `json:"index_name"`
	Timestamp  int64   `json:"timestamp"`
}

// DeribitClient manages the Deribit WebSocket connection for DVOL.
type DeribitClient struct {
	conn *Conn
	log  zerolog.Logger
	bus  *eventbus.Bus
}

// SetEventBus sets the event bus for publishing parsed events.
func (d *DeribitClient) SetEventBus(bus *eventbus.Bus) {
	d.bus = bus
}

// NewDeribitClient creates a new Deribit WS client.
func NewDeribitClient(logger zerolog.Logger) *DeribitClient {
	d := &DeribitClient{
		log: logger.With().Str("source", "deribit").Logger(),
	}
	d.conn = NewConn(deribitURL, nil, d.handleMessage, d.log)
	d.conn.SetOnConnect(d.onConnect)
	return d
}

// IsConnected returns whether the WebSocket is currently active.
func (d *DeribitClient) IsConnected() bool {
	return d.conn != nil && d.conn.IsConnected()
}

// Connect starts the Deribit WebSocket connection.
func (d *DeribitClient) Connect() error {
	return d.conn.Connect()
}

// Disconnect stops the Deribit WebSocket connection.
func (d *DeribitClient) Disconnect() {
	d.conn.Disconnect()
}

func (d *DeribitClient) onConnect(c *Conn) error {
	req := DeribitRPCRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "public/subscribe",
		Params: map[string]interface{}{
			"channels": []string{"deribit_volatility_index.btc_usd"},
		},
	}
	return c.WriteJSON(req)
}

func (d *DeribitClient) handleMessage(_ int, data []byte) {
	var resp DeribitRPCResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		d.log.Error().Err(err).Msg("failed to parse deribit response")
		return
	}

	// Subscription confirmation
	if resp.ID != nil {
		d.log.Info().Int("id", *resp.ID).Msg("deribit subscription confirmed")
		return
	}

	// Subscription notification
	if resp.Method == "subscription" && resp.Params != nil {
		var sub DeribitSubscriptionData
		if err := json.Unmarshal(resp.Params, &sub); err != nil {
			d.log.Error().Err(err).Msg("failed to parse deribit subscription data")
			return
		}

		if sub.Channel == "deribit_volatility_index.btc_usd" {
			var dvol DeribitDVOL
			if err := json.Unmarshal(sub.Data, &dvol); err != nil {
				d.log.Error().Err(err).Msg("failed to parse dvol data")
				return
			}
			if d.bus != nil {
				d.bus.Publish(eventbus.DeribitDVOL{
					Value: dvol.Volatility, Timestamp: dvol.Timestamp,
				})
			}

			d.log.Info().
				Float64("dvol", dvol.Volatility).
				Int64("ts", dvol.Timestamp).
				Msg("dvol update")
		}
	}
}
