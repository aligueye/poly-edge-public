package config

import (
	"fmt"
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

type Config struct {
	PolyPrivateKey      string
	PolyApiKey          string
	PolyApiSecret       string
	PolyApiPassphrase   string
	PolyChainID         int
	PrivateKey          string
	WalletAddress       string
	BankrollUSDC        float64
	PaperTrade          bool
	LogLevel            string
	MetricsPort         int
	GrpcSocketPath      string
	TelegramBotToken    string
	TelegramChatID      string
	Socks5ProxyURL      string
	MarketMakerEnabled  bool
}

func Load() (*Config, error) {
	_ = godotenv.Load()

	cfg := &Config{
		PolyPrivateKey:    os.Getenv("POLY_PRIVATE_KEY"),
		PolyApiKey:        os.Getenv("POLY_API_KEY"),
		PolyApiSecret:     os.Getenv("POLY_API_SECRET"),
		PolyApiPassphrase: os.Getenv("POLY_API_PASSPHRASE"),
		PolyChainID:       137,
		BankrollUSDC:      0,
		PaperTrade:        true,
		LogLevel:          "info",
		MetricsPort:       9090,
		MarketMakerEnabled: true,
		GrpcSocketPath:    "/tmp/polyedge.sock",
	}

	if v := os.Getenv("POLY_CHAIN_ID"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return nil, fmt.Errorf("POLY_CHAIN_ID: %w", err)
		}
		cfg.PolyChainID = n
	}

	if v := os.Getenv("BANKROLL_USDC"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return nil, fmt.Errorf("BANKROLL_USDC: %w", err)
		}
		cfg.BankrollUSDC = f
	}

	if v := os.Getenv("PAPER_TRADE"); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return nil, fmt.Errorf("PAPER_TRADE: %w", err)
		}
		cfg.PaperTrade = b
	}

	if v := os.Getenv("MARKET_MAKER_ENABLED"); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return nil, fmt.Errorf("MARKET_MAKER_ENABLED: %w", err)
		}
		cfg.MarketMakerEnabled = b
	}

	if v := os.Getenv("LOG_LEVEL"); v != "" {
		cfg.LogLevel = v
	}

	if v := os.Getenv("METRICS_PORT"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return nil, fmt.Errorf("METRICS_PORT: %w", err)
		}
		cfg.MetricsPort = n
	}

	cfg.PrivateKey = os.Getenv("PRIVATE_KEY")
	cfg.WalletAddress = os.Getenv("WALLET_ADDRESS")

	if v := os.Getenv("GRPC_SOCKET_PATH"); v != "" {
		cfg.GrpcSocketPath = v
	}

	cfg.TelegramBotToken = os.Getenv("TELEGRAM_BOT_TOKEN")
	cfg.TelegramChatID = os.Getenv("TELEGRAM_CHAT_ID")

	// Build SOCKS5 proxy URL if configured
	if host := os.Getenv("SOCKS5_PROXY_HOST"); host != "" {
		port := os.Getenv("SOCKS5_PROXY_PORT")
		user := os.Getenv("SOCKS5_PROXY_USER")
		pass := os.Getenv("SOCKS5_PROXY_PASS")
		if user != "" && pass != "" {
			cfg.Socks5ProxyURL = fmt.Sprintf("socks5://%s:%s@%s:%s", user, pass, host, port)
		} else {
			cfg.Socks5ProxyURL = fmt.Sprintf("socks5://%s:%s", host, port)
		}
	}

	if err := cfg.validate(); err != nil {
		return nil, err
	}

	return cfg, nil
}

func (c *Config) validate() error {
	if c.PolyPrivateKey == "" {
		return fmt.Errorf("POLY_PRIVATE_KEY is required")
	}
	if c.BankrollUSDC <= 0 {
		return fmt.Errorf("BANKROLL_USDC must be > 0")
	}
	if !c.PaperTrade {
		if c.PrivateKey == "" {
			return fmt.Errorf("PRIVATE_KEY is required for live trading")
		}
		if c.WalletAddress == "" {
			return fmt.Errorf("WALLET_ADDRESS is required for live trading")
		}
		if c.PolyApiKey == "" || c.PolyApiSecret == "" || c.PolyApiPassphrase == "" {
			return fmt.Errorf("POLY_API_KEY, POLY_API_SECRET, POLY_API_PASSPHRASE are required for live trading")
		}
	}
	return nil
}
