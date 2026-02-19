package metrics

import (
	"fmt"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/rs/zerolog"
)

// Metrics exposes Prometheus metrics for the trading system.
type Metrics struct {
	// Orders
	OrdersTotal    *prometheus.CounterVec
	OrderLatency   *prometheus.HistogramVec
	OrderErrors    *prometheus.CounterVec
	OrdersCanceled prometheus.Counter

	// Position & PnL
	FillRate          *prometheus.GaugeVec
	PnLRealized       *prometheus.GaugeVec
	PnLUnrealized     *prometheus.GaugeVec
	PositionExposure  *prometheus.GaugeVec
	DrawdownCurrent   prometheus.Gauge
	BankrollAvailable prometheus.Gauge

	// WebSocket
	WSConnected *prometheus.GaugeVec

	// Directional strategy
	DirectionalAccuracy   prometheus.Gauge
	DirectionalConfidence prometheus.Gauge

	// Market maker
	MMInventory *prometheus.GaugeVec
	MMSpreadBps prometheus.Gauge

	log zerolog.Logger
}

// New creates and registers all Prometheus metrics.
func New(logger zerolog.Logger) *Metrics {
	m := &Metrics{
		log: logger.With().Str("component", "metrics").Logger(),

		OrdersTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "polyedge",
			Name:      "orders_total",
			Help:      "Total orders by strategy, side, and status",
		}, []string{"strategy", "side", "status"}),

		OrderLatency: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "polyedge",
			Name:      "order_latency_seconds",
			Help:      "Order submission latency in seconds",
			Buckets:   []float64{0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0},
		}, []string{"strategy"}),

		OrderErrors: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "polyedge",
			Name:      "order_errors_total",
			Help:      "Total order submission errors",
		}, []string{"reason"}),

		OrdersCanceled: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: "polyedge",
			Name:      "orders_canceled_total",
			Help:      "Total orders canceled",
		}),

		FillRate: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "polyedge",
			Name:      "fill_rate",
			Help:      "Fill rate per strategy",
		}, []string{"strategy"}),

		PnLRealized: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "polyedge",
			Name:      "pnl_realized",
			Help:      "Realized PnL per strategy",
		}, []string{"strategy"}),

		PnLUnrealized: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "polyedge",
			Name:      "pnl_unrealized",
			Help:      "Unrealized PnL per strategy",
		}, []string{"strategy"}),

		PositionExposure: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "polyedge",
			Name:      "position_exposure",
			Help:      "Position exposure per strategy and side",
		}, []string{"strategy", "side"}),

		DrawdownCurrent: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "polyedge",
			Name:      "drawdown_current",
			Help:      "Current drawdown as fraction of bankroll",
		}),

		BankrollAvailable: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "polyedge",
			Name:      "bankroll_available",
			Help:      "Available bankroll in USDC",
		}),

		WSConnected: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "polyedge",
			Name:      "ws_connected",
			Help:      "WebSocket connection status (1=connected, 0=disconnected)",
		}, []string{"source"}),

		DirectionalAccuracy: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "polyedge",
			Name:      "directional_accuracy",
			Help:      "Rolling directional prediction accuracy",
		}),

		DirectionalConfidence: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "polyedge",
			Name:      "directional_confidence_avg",
			Help:      "Average confidence of directional predictions",
		}),

		MMInventory: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "polyedge",
			Name:      "mm_inventory",
			Help:      "Market maker inventory per side",
		}, []string{"side"}),

		MMSpreadBps: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "polyedge",
			Name:      "mm_spread_bps",
			Help:      "Market maker quoted spread in basis points",
		}),
	}

	prometheus.MustRegister(
		m.OrdersTotal,
		m.OrderLatency,
		m.OrderErrors,
		m.OrdersCanceled,
		m.FillRate,
		m.PnLRealized,
		m.PnLUnrealized,
		m.PositionExposure,
		m.DrawdownCurrent,
		m.BankrollAvailable,
		m.WSConnected,
		m.DirectionalAccuracy,
		m.DirectionalConfidence,
		m.MMInventory,
		m.MMSpreadBps,
	)

	return m
}

// NewNoop creates unregistered metrics suitable for testing.
func NewNoop() *Metrics {
	return &Metrics{
		OrdersTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "test_orders_total",
		}, []string{"strategy", "side", "status"}),
		OrderLatency: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name: "test_order_latency",
		}, []string{"strategy"}),
		OrderErrors: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "test_order_errors",
		}, []string{"reason"}),
		OrdersCanceled: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "test_orders_canceled",
		}),
		FillRate:          prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "test_fill_rate"}, []string{"strategy"}),
		PnLRealized:       prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "test_pnl_realized"}, []string{"strategy"}),
		PnLUnrealized:     prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "test_pnl_unrealized"}, []string{"strategy"}),
		PositionExposure:  prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "test_position_exposure"}, []string{"strategy", "side"}),
		DrawdownCurrent:   prometheus.NewGauge(prometheus.GaugeOpts{Name: "test_drawdown"}),
		BankrollAvailable: prometheus.NewGauge(prometheus.GaugeOpts{Name: "test_bankroll"}),
		WSConnected:       prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "test_ws_connected"}, []string{"source"}),
		DirectionalAccuracy:   prometheus.NewGauge(prometheus.GaugeOpts{Name: "test_dir_accuracy"}),
		DirectionalConfidence: prometheus.NewGauge(prometheus.GaugeOpts{Name: "test_dir_confidence"}),
		MMInventory: prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: "test_mm_inventory"}, []string{"side"}),
		MMSpreadBps: prometheus.NewGauge(prometheus.GaugeOpts{Name: "test_mm_spread"}),
	}
}

// Serve starts the Prometheus HTTP metrics server on the given port.
func (m *Metrics) Serve(port int) {
	addr := fmt.Sprintf(":%d", port)
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	m.log.Info().Str("addr", addr).Msg("starting metrics server")
	go func() {
		if err := http.ListenAndServe(addr, mux); err != nil {
			m.log.Error().Err(err).Msg("metrics server error")
		}
	}()
}
