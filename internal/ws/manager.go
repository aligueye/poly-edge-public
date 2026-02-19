package ws

import (
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/rs/zerolog"
)

const (
	initialBackoff = 1 * time.Second
	maxBackoff     = 30 * time.Second
	pongWait       = 60 * time.Second
	pingInterval   = 30 * time.Second
	writeWait      = 10 * time.Second
)

// MessageHandler is called for each message received on the connection.
type MessageHandler func(msgType int, data []byte)

// OnConnectHook is called after a successful (re)connection.
type OnConnectHook func(c *Conn) error

// Conn is a generic WebSocket connection wrapper with auto-reconnect.
type Conn struct {
	url       string
	header    http.Header
	handler   MessageHandler
	onConnect OnConnectHook
	log       zerolog.Logger

	mu        sync.RWMutex
	conn      *websocket.Conn
	connected bool

	writeMu sync.Mutex // serializes all writes to the websocket

	done chan struct{}
	wg   sync.WaitGroup
}

// NewConn creates a new managed WebSocket connection.
func NewConn(url string, header http.Header, handler MessageHandler, logger zerolog.Logger) *Conn {
	return &Conn{
		url:     url,
		header:  header,
		handler: handler,
		log:     logger,
		done:    make(chan struct{}),
	}
}

// SetOnConnect sets a hook that runs after each successful connection.
func (c *Conn) SetOnConnect(hook OnConnectHook) {
	c.onConnect = hook
}

// Connect dials the WebSocket and starts the read loop.
func (c *Conn) Connect() error {
	if err := c.dial(); err != nil {
		c.log.Warn().Err(err).Msg("initial connection failed, starting reconnect loop")
		c.wg.Add(1)
		go c.reconnectLoop()
		return nil
	}
	c.startReadLoop()
	return nil
}

// Disconnect closes the connection and stops reconnection.
func (c *Conn) Disconnect() {
	close(c.done)
	c.mu.Lock()
	if c.conn != nil {
		c.writeMu.Lock()
		_ = c.conn.WriteMessage(
			websocket.CloseMessage,
			websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""),
		)
		c.writeMu.Unlock()
		_ = c.conn.Close()
		c.connected = false
	}
	c.mu.Unlock()
	c.wg.Wait()
}

// IsConnected returns whether the connection is currently active.
func (c *Conn) IsConnected() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.connected
}

// WriteMessage sends a message on the connection, serialized with a mutex
// to prevent concurrent write panics.
func (c *Conn) WriteMessage(msgType int, data []byte) error {
	c.mu.RLock()
	ws := c.conn
	c.mu.RUnlock()
	if ws == nil {
		return nil
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	_ = ws.SetWriteDeadline(time.Now().Add(writeWait))
	return ws.WriteMessage(msgType, data)
}

// WriteJSON sends a JSON-encoded message on the connection, serialized
// with a mutex to prevent concurrent write panics.
func (c *Conn) WriteJSON(v interface{}) error {
	c.mu.RLock()
	ws := c.conn
	c.mu.RUnlock()
	if ws == nil {
		return nil
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	_ = ws.SetWriteDeadline(time.Now().Add(writeWait))
	return ws.WriteJSON(v)
}

func (c *Conn) dial() error {
	ws, _, err := websocket.DefaultDialer.Dial(c.url, c.header)
	if err != nil {
		return err
	}

	ws.SetPongHandler(func(string) error {
		return ws.SetReadDeadline(time.Now().Add(pongWait))
	})

	c.mu.Lock()
	c.conn = ws
	c.connected = true
	c.mu.Unlock()

	if c.onConnect != nil {
		if err := c.onConnect(c); err != nil {
			c.log.Error().Err(err).Msg("on-connect hook failed")
			c.mu.Lock()
			_ = ws.Close()
			c.connected = false
			c.conn = nil
			c.mu.Unlock()
			return err
		}
	}

	c.log.Info().Str("url", c.url).Msg("websocket connected")
	return nil
}

func (c *Conn) startReadLoop() {
	c.wg.Add(2)
	go c.readLoop()
	go c.pingLoop()
}

func (c *Conn) readLoop() {
	defer c.wg.Done()
	for {
		c.mu.RLock()
		ws := c.conn
		c.mu.RUnlock()
		if ws == nil {
			return
		}

		_ = ws.SetReadDeadline(time.Now().Add(pongWait))
		msgType, data, err := ws.ReadMessage()
		if err != nil {
			select {
			case <-c.done:
				return
			default:
			}
			c.log.Warn().Err(err).Msg("websocket read error, reconnecting")
			c.mu.Lock()
			c.connected = false
			_ = ws.Close()
			c.conn = nil
			c.mu.Unlock()
			c.wg.Add(1)
			go c.reconnectLoop()
			return
		}

		c.handler(msgType, data)
	}
}

func (c *Conn) pingLoop() {
	defer c.wg.Done()
	ticker := time.NewTicker(pingInterval)
	defer ticker.Stop()
	for {
		select {
		case <-c.done:
			return
		case <-ticker.C:
			if err := c.WriteMessage(websocket.PingMessage, nil); err != nil {
				c.log.Warn().Err(err).Msg("ping failed")
				return
			}
		}
	}
}

func (c *Conn) reconnectLoop() {
	defer c.wg.Done()
	attempt := 0
	for {
		select {
		case <-c.done:
			return
		default:
		}

		backoff := time.Duration(math.Pow(2, float64(attempt))) * time.Second
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
		c.log.Warn().Dur("backoff", backoff).Int("attempt", attempt+1).Msg("reconnecting")

		select {
		case <-c.done:
			return
		case <-time.After(backoff):
		}

		if err := c.dial(); err != nil {
			c.log.Warn().Err(err).Msg("reconnect attempt failed")
			attempt++
			continue
		}

		c.log.Info().Int("after_attempts", attempt+1).Msg("reconnected")
		c.startReadLoop()
		return
	}
}
