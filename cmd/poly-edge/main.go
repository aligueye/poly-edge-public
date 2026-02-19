package main

import (
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ali/poly-edge/internal/config"
	"github.com/ali/poly-edge/internal/eventbus"
	"github.com/ali/poly-edge/internal/execution"
	polygrpc "github.com/ali/poly-edge/internal/grpc"
	"github.com/ali/poly-edge/internal/metrics"
	"github.com/ali/poly-edge/internal/position"
	"github.com/ali/poly-edge/internal/risk"
	"github.com/ali/poly-edge/internal/strategy"
	"github.com/ali/poly-edge/internal/ta"
	"github.com/ali/poly-edge/internal/ws"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	cfg, err := config.Load()
	if err != nil {
		log.Fatal().Err(err).Msg("failed to load config")
	}

	lvl, err := zerolog.ParseLevel(cfg.LogLevel)
	if err != nil {
		lvl = zerolog.InfoLevel
	}
	zerolog.SetGlobalLevel(lvl)

	log.Info().
		Float64("bankroll", cfg.BankrollUSDC).
		Bool("paper", cfg.PaperTrade).
		Msg("poly-edge starting")

	// Event bus
	bus := eventbus.New(log.Logger)

	// Metrics
	m := metrics.New(log.Logger)
	m.Serve(cfg.MetricsPort)

	// Position manager
	posManager := position.NewManager(cfg.BankrollUSDC, log.Logger)

	// Risk engine
	riskEngine := risk.NewEngine(cfg.BankrollUSDC)

	// Executor
	var clob *execution.ClobClient
	if !cfg.PaperTrade {
		clob, err = execution.NewClobClient(execution.ClobConfig{
			APIKey:        cfg.PolyApiKey,
			APISecret:     cfg.PolyApiSecret,
			APIPassphrase: cfg.PolyApiPassphrase,
			PrivateKey:    cfg.PrivateKey,
			WalletAddress: cfg.WalletAddress,
			ChainID:       cfg.PolyChainID,
			ProxyURL:      cfg.Socks5ProxyURL,
		}, log.Logger)
		if err != nil {
			log.Fatal().Err(err).Msg("failed to create clob client")
		}
	}
	executor := execution.NewExecutor(riskEngine, posManager, m, clob, cfg.PaperTrade, log.Logger)

	// Market discovery
	discovery := execution.NewMarketDiscovery(log.Logger)

	// TA engine
	taEngine := ta.NewEngine(log.Logger)
	go taEngine.Run(bus)

	// gRPC signal client (connects to Python sidecar)
	signalClient, err := polygrpc.NewSignalClient(cfg.GrpcSocketPath, taEngine, discovery, log.Logger)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to create signal client")
	}
	signalClient.SubscribeToEvents(bus)
	go signalClient.WaitForSidecar(cfg.GrpcSocketPath)

	// Market resolver — determines YES/NO winner after each 5-min window
	resolver := execution.NewResolver(posManager, signalClient.GetBTCPrice, log.Logger)

	// Strategy router (directional + market maker)
	router := strategy.NewRouter(signalClient, executor, discovery, posManager, m, bus, cfg.BankrollUSDC, cfg.MarketMakerEnabled, log.Logger)

	// Wire resolution callback: resolver → router (accuracy tracking)
	resolver.OnResolution(func(conditionID, winnerSide string) {
		router.RecordResolution(conditionID, winnerSide)
	})

	// Wire discovery → resolver (track new market windows)
	discovery.OnChange(func(market execution.MarketInfo) {
		resolver.TrackMarket(market.ConditionID, market.EndTime, signalClient.GetBTCPrice())
		log.Info().
			Str("condition_id", market.ConditionID).
			Str("slug", market.EventSlug).
			Msg("active market updated")
	})
	go discovery.Run()

	// Start executor (P2: cancel orphaned orders, then launch fill poller)
	executor.Start()
	go resolver.Run()
	go router.Run()

	// WebSocket clients
	binance := ws.NewBinanceClient(log.Logger)
	binance.SetEventBus(bus)

	poly := ws.NewPolymarketClient(log.Logger)
	poly.SetEventBus(bus)

	deribit := ws.NewDeribitClient(log.Logger)
	deribit.SetEventBus(bus)

	// Start all connections in separate goroutines
	go func() {
		if err := binance.Connect(); err != nil {
			log.Error().Err(err).Msg("binance connect failed")
		}
	}()
	go func() {
		if err := poly.Connect(); err != nil {
			log.Error().Err(err).Msg("polymarket connect failed")
		}
	}()
	go func() {
		if err := deribit.Connect(); err != nil {
			log.Error().Err(err).Msg("deribit connect failed")
		}
	}()

	// Metrics ticker — polls component state every 5s for Prometheus gauges
	metricsDone := make(chan struct{})
	go func() {
		boolToFloat := func(b bool) float64 {
			if b {
				return 1
			}
			return 0
		}
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				// WebSocket status
				m.WSConnected.WithLabelValues("binance").Set(boolToFloat(binance.IsConnected()))
				m.WSConnected.WithLabelValues("polymarket").Set(boolToFloat(poly.IsConnected()))
				m.WSConnected.WithLabelValues("deribit").Set(boolToFloat(deribit.IsConnected()))

				// Portfolio state
				p := posManager.GetPortfolio()
				m.BankrollAvailable.Set(p.Available)
				m.DrawdownCurrent.Set(p.DailyPnL / p.TotalBankroll)
				m.PnLRealized.WithLabelValues("directional").Set(p.RealizedPnL)
				m.PositionExposure.WithLabelValues("directional", "total").Set(p.Deployed)
			case <-metricsDone:
				return
			}
		}
	}()

	// Block on signal for graceful shutdown
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig

	close(metricsDone)

	log.Info().Msg("shutting down...")

	executor.Stop()
	resolver.Stop()
	router.Stop()
	signalClient.Stop()
	discovery.Stop()
	taEngine.Stop()
	binance.Disconnect()
	poly.Disconnect()
	deribit.Disconnect()

	// Session summary
	portfolio := posManager.GetPortfolio()
	total, correct, avgConf := router.DirectionalStats()
	log.Info().Msg("═══════════════════════════════════════")
	log.Info().Msg("           SESSION SUMMARY")
	log.Info().Msg("═══════════════════════════════════════")
	log.Info().
		Float64("realized_pnl", portfolio.RealizedPnL).
		Float64("daily_pnl", portfolio.DailyPnL).
		Float64("deployed", portfolio.Deployed).
		Float64("available", portfolio.Available).
		Msg("portfolio")
	if total > 0 {
		log.Info().
			Int("resolved", total).
			Int("correct", correct).
			Float64("accuracy", float64(correct)/float64(total)).
			Float64("avg_confidence", avgConf).
			Msg("directional")
	} else {
		log.Info().Msg("directional: no resolutions yet")
	}
	log.Info().Msg("═══════════════════════════════════════")
	log.Info().Msg("poly-edge stopped")
}
