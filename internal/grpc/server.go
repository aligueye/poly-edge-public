package grpc

import (
	"context"
	"os"
	"sync"
	"time"

	"github.com/ali/poly-edge/internal/eventbus"
	"github.com/ali/poly-edge/internal/execution"
	pb "github.com/ali/poly-edge/internal/grpc/proto"
	"github.com/ali/poly-edge/internal/ta"
	"github.com/rs/zerolog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	maxRecentTrades = 200
	maxRecentLiqs   = 50
	minCallInterval = 500 * time.Millisecond // max 2 calls/sec
)

// SignalClient wraps a gRPC client connection to the Python sidecar.
// It caches latest market data from the event bus and assembles
// MarketState for each GetSignal call.
type SignalClient struct {
	conn   *grpc.ClientConn
	client pb.SignalServiceClient

	// Data sources
	taEngine  *ta.Engine
	discovery *execution.MarketDiscovery

	// Cached event bus data
	mu          sync.RWMutex
	btcPrice    float64
	book        eventbus.BinanceBookUpdate
	polyYes     float64
	polyNo      float64
	dvol        float64
	fundingRate float64
	trades      []eventbus.BinanceTrade
	liqs        []eventbus.BinanceLiquidation

	// Rate limiting
	lastCall time.Time

	log  zerolog.Logger
	done chan struct{}
}

// NewSignalClient creates a client that connects to the Python sidecar over UDS.
func NewSignalClient(
	socketPath string,
	taEngine *ta.Engine,
	discovery *execution.MarketDiscovery,
	logger zerolog.Logger,
) (*SignalClient, error) {
	sc := &SignalClient{
		taEngine:  taEngine,
		discovery: discovery,
		trades:    make([]eventbus.BinanceTrade, 0, maxRecentTrades),
		liqs:      make([]eventbus.BinanceLiquidation, 0, maxRecentLiqs),
		log:       logger.With().Str("component", "grpc-client").Logger(),
		done:      make(chan struct{}),
	}

	return sc, nil
}

// Connect dials the Python sidecar's UDS. Call this after the sidecar is up.
func (sc *SignalClient) Connect(socketPath string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, "unix://"+socketPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return err
	}
	sc.conn = conn
	sc.client = pb.NewSignalServiceClient(conn)
	sc.log.Info().Str("socket", socketPath).Msg("connected to python sidecar")
	return nil
}

// WaitForSidecar waits until the sidecar socket exists, then connects.
func (sc *SignalClient) WaitForSidecar(socketPath string) {
	sc.log.Info().Str("socket", socketPath).Msg("waiting for python sidecar...")
	for {
		select {
		case <-sc.done:
			return
		case <-time.After(2 * time.Second):
			if _, err := os.Stat(socketPath); err == nil {
				if err := sc.Connect(socketPath); err != nil {
					sc.log.Warn().Err(err).Msg("sidecar socket exists but connect failed")
					continue
				}
				return
			}
		}
	}
}

// SubscribeToEvents subscribes to event bus channels and caches latest data.
func (sc *SignalClient) SubscribeToEvents(bus *eventbus.Bus) {
	bookCh := bus.Subscribe(eventbus.EventBinanceBookUpdate)
	tradeCh := bus.Subscribe(eventbus.EventBinanceTrade)
	liqCh := bus.Subscribe(eventbus.EventBinanceLiquidation)
	markCh := bus.Subscribe(eventbus.EventBinanceMarkPrice)
	polyCh := bus.Subscribe(eventbus.EventPolymarketBookUpdate)
	priceCh := bus.Subscribe(eventbus.EventPolymarketPrice)
	dvolCh := bus.Subscribe(eventbus.EventDeribitDVOL)

	go sc.cacheLoop(bookCh, tradeCh, liqCh, markCh, polyCh, priceCh, dvolCh)
}

func (sc *SignalClient) cacheLoop(
	bookCh, tradeCh, liqCh, markCh, polyCh, priceCh, dvolCh <-chan eventbus.Event,
) {
	for {
		select {
		case <-sc.done:
			return
		case evt := <-bookCh:
			book := evt.(eventbus.BinanceBookUpdate)
			sc.mu.Lock()
			sc.book = book
			sc.mu.Unlock()
		case evt := <-tradeCh:
			trade := evt.(eventbus.BinanceTrade)
			sc.mu.Lock()
			sc.btcPrice = trade.Price
			sc.trades = append(sc.trades, trade)
			if len(sc.trades) > maxRecentTrades {
				sc.trades = sc.trades[len(sc.trades)-maxRecentTrades:]
			}
			sc.mu.Unlock()
		case evt := <-liqCh:
			liq := evt.(eventbus.BinanceLiquidation)
			sc.mu.Lock()
			sc.liqs = append(sc.liqs, liq)
			if len(sc.liqs) > maxRecentLiqs {
				sc.liqs = sc.liqs[len(sc.liqs)-maxRecentLiqs:]
			}
			sc.mu.Unlock()
		case evt := <-markCh:
			mark := evt.(eventbus.BinanceMarkPrice)
			sc.mu.Lock()
			sc.fundingRate = mark.FundingRate
			sc.mu.Unlock()
		case evt := <-polyCh:
			poly := evt.(eventbus.PolymarketBookUpdate)
			sc.mu.Lock()
			sc.polyYes = poly.YesBestBid
			sc.polyNo = poly.NoBestBid
			sc.mu.Unlock()
		case evt := <-priceCh:
			price := evt.(eventbus.PolymarketPrice)
			sc.mu.Lock()
			sc.btcPrice = price.Price
			sc.mu.Unlock()
		case evt := <-dvolCh:
			d := evt.(eventbus.DeribitDVOL)
			sc.mu.Lock()
			sc.dvol = d.Value
			sc.mu.Unlock()
		}
	}
}

// GetSignal assembles MarketState from cached data and calls the Python sidecar.
func (sc *SignalClient) GetSignal(ctx context.Context) (*pb.Signal, error) {
	if sc.client == nil {
		return &pb.Signal{Action: "hold", Confidence: 0}, nil
	}

	// Rate limiting: max 2 calls/sec
	sc.mu.Lock()
	since := time.Since(sc.lastCall)
	if since < minCallInterval {
		sc.mu.Unlock()
		time.Sleep(minCallInterval - since)
		sc.mu.Lock()
	}
	sc.lastCall = time.Now()

	state := sc.buildMarketState()
	sc.mu.Unlock()

	signal, err := sc.client.GetSignal(ctx, state)
	if err != nil {
		sc.log.Warn().Err(err).Msg("sidecar GetSignal failed")
		return &pb.Signal{Action: "hold", Confidence: 0}, err
	}

	sc.log.Debug().
		Str("action", signal.Action).
		Float64("confidence", signal.Confidence).
		Msg("signal received")

	return signal, nil
}

// buildMarketState assembles the proto message from cached data.
// Caller must hold sc.mu (at least read lock).
func (sc *SignalClient) buildMarketState() *pb.MarketState {
	snap := sc.taEngine.GetIndicators()

	// Flatten order book
	bids := make([]float64, 0, len(sc.book.Bids)*2)
	for _, lvl := range sc.book.Bids {
		bids = append(bids, lvl.Price, lvl.Qty)
	}
	asks := make([]float64, 0, len(sc.book.Asks)*2)
	for _, lvl := range sc.book.Asks {
		asks = append(asks, lvl.Price, lvl.Qty)
	}

	// Recent trades
	trades := make([]*pb.AggTrade, len(sc.trades))
	for i, t := range sc.trades {
		trades[i] = &pb.AggTrade{
			Price:        t.Price,
			Quantity:     t.Qty,
			IsBuyerMaker: t.IsBuyerMaker,
			TimestampMs:  t.Timestamp,
		}
	}

	// Recent liquidations
	liqs := make([]*pb.Liquidation, len(sc.liqs))
	for i, l := range sc.liqs {
		liqs[i] = &pb.Liquidation{
			Side:        l.Side,
			Quantity:    l.Qty,
			Price:       l.Price,
			TimestampMs: l.Timestamp,
		}
	}

	// Window timing from market discovery
	var windowOpen float64
	var secsRemaining float64
	if mkt := sc.discovery.GetCurrent(); mkt != nil {
		secsRemaining = time.Until(mkt.EndTime).Seconds()
		if secsRemaining < 0 {
			secsRemaining = 0
		}
		// Use the BTC price at start of window approximated by open price
		// (will be set properly once directional strategy tracks it)
		windowOpen = sc.btcPrice
	}

	return &pb.MarketState{
		BtcPrice:             sc.btcPrice,
		WindowOpenPrice:      windowOpen,
		SecondsRemaining:     secsRemaining,
		PolymarketYesPrice:   sc.polyYes,
		PolymarketNoPrice:    sc.polyNo,
		Dvol:                 sc.dvol,
		BinanceBids:          bids,
		BinanceAsks:          asks,
		RecentTrades:         trades,
		RecentLiquidations:   liqs,
		FundingRate:          sc.fundingRate,
		Rsi_7:               snap.RSI7,
		Rsi_14:              snap.RSI14,
		Rsi_30:              snap.RSI30,
		MacdSignal:          snap.MACDSignal,
		StochK:              snap.StochK,
		Momentum_30S:        snap.Momentum30s,
		Momentum_60S:        snap.Momentum60s,
		Momentum_120S:       snap.Momentum120s,
		Ema_9Vs_21:          snap.EMA9vsEMA21,
		Atr_14:              snap.ATR14,
		BollingerPctB:       snap.BollingerPctB,
		HourlyTrend:         snap.HourlyTrend,
	}
}

// Connected returns true if the gRPC client is connected.
func (sc *SignalClient) Connected() bool {
	return sc.client != nil
}

// GetBTCPrice returns the latest cached BTC/USD price.
func (sc *SignalClient) GetBTCPrice() float64 {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return sc.btcPrice
}

// Stop shuts down the gRPC client connection.
func (sc *SignalClient) Stop() {
	close(sc.done)
	if sc.conn != nil {
		sc.conn.Close()
	}
}

