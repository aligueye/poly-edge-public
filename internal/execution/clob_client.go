package execution

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/rs/zerolog"
)

const (
	clobBaseURL = "https://clob.polymarket.com"
)

// ClobConfig holds credentials for the Polymarket CLOB API.
type ClobConfig struct {
	BaseURL       string
	APIKey        string
	APISecret     string
	APIPassphrase string
	PrivateKey    string // hex-encoded ECDSA private key
	WalletAddress string
	ChainID       int
	ProxyURL      string // SOCKS5 proxy URL (optional)
}

// ClobOrderResponse is the response from POST /order.
type ClobOrderResponse struct {
	Success     bool     `json:"success"`
	ErrorMsg    string   `json:"errorMsg"`
	OrderID     string   `json:"orderId"`
	OrderHashes []string `json:"orderHashes"`
	Status      string   `json:"status"`
}

// ClobCancelResponse is the response from cancel endpoints.
type ClobCancelResponse struct {
	Canceled    []string          `json:"canceled"`
	NotCanceled map[string]string `json:"not_canceled"`
}

// ClobOrderStatusResp is the response from GET /data/order/<id>.
type ClobOrderStatusResp struct {
	ID           string `json:"id"`
	Status       string `json:"status"`
	SizeMatched  string `json:"size_matched"`
	Price        string `json:"price"`
	OriginalSize string `json:"original_size"`
	Side         string `json:"side"`
	TokenID      string `json:"asset_id"`
	Market       string `json:"market"`
}

// feeRateResp is the response from GET /fee-rate.
type feeRateResp struct {
	BaseFee int `json:"base_fee"`
}

// negRiskResp is the response from GET /neg-risk.
type negRiskResp struct {
	NegRisk bool `json:"neg_risk"`
}

// tickSizeResp is the response from GET /tick-size.
type tickSizeResp struct {
	MinimumTickSize float64 `json:"minimum_tick_size"`
}

// ClobClient handles HTTP communication with the Polymarket CLOB API.
type ClobClient struct {
	config        ClobConfig
	signer        *OrderSigner
	wallet        common.Address
	http          *http.Client
	log           zerolog.Logger
	tickSizeCache map[string]string // tokenID → tick size string
}

// NewClobClient creates a CLOB API client with EIP-712 signing.
func NewClobClient(cfg ClobConfig, logger zerolog.Logger) (*ClobClient, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = clobBaseURL
	}
	if cfg.ChainID == 0 {
		cfg.ChainID = 137
	}

	// Parse private key (strip 0x prefix if present)
	pkHex := strings.TrimPrefix(cfg.PrivateKey, "0x")
	key, err := crypto.HexToECDSA(pkHex)
	if err != nil {
		return nil, fmt.Errorf("invalid private key: %w", err)
	}

	signer, err := NewOrderSigner(key, cfg.ChainID)
	if err != nil {
		return nil, err
	}

	wallet := common.HexToAddress(cfg.WalletAddress)

	log := logger.With().Str("component", "clob").Logger()
	log.Info().
		Str("signer", signer.Address().Hex()).
		Str("wallet", wallet.Hex()).
		Int("chain_id", cfg.ChainID).
		Msg("clob client initialized")

	httpClient := &http.Client{Timeout: 10 * time.Second}
	if cfg.ProxyURL != "" {
		proxyURL, err := url.Parse(cfg.ProxyURL)
		if err != nil {
			return nil, fmt.Errorf("invalid proxy URL: %w", err)
		}
		httpClient.Transport = &http.Transport{Proxy: http.ProxyURL(proxyURL)}
		log.Info().Str("proxy", proxyURL.Host).Msg("using SOCKS5 proxy for CLOB")
	}

	return &ClobClient{
		config:        cfg,
		signer:        signer,
		wallet:        wallet,
		http:          httpClient,
		log:           log,
		tickSizeCache: make(map[string]string),
	}, nil
}

// PlaceOrder builds, signs, and submits an order to the Polymarket CLOB.
func (c *ClobClient) PlaceOrder(req OrderRequest) (ClobOrderResponse, error) {
	postOnly := req.OrderType == OrderTypePostOnly
	orderType := string(req.OrderType)
	if postOnly {
		orderType = "GTC"
	}

	// Fetch fee rate, neg_risk, and tick size for this token from the CLOB API
	feeRate, err := c.getFeeRate(req.TokenID)
	if err != nil {
		c.log.Warn().Err(err).Msg("failed to fetch fee rate, using default 0")
		feeRate = 0
	}
	negRisk := c.getNegRisk(req.TokenID)
	tickSize := c.getTickSize(req.TokenID)

	// Compute amounts with tick-size-aware rounding (price, size, and product)
	isFOK := req.OrderType == OrderTypeFOK || req.OrderType == OrderTypeFAK
	makerAmt, takerAmt := ComputeAmounts(req.Price, req.Size, req.Side, tickSize, isFOK)

	// Parse token ID as big.Int
	tokenID := new(big.Int)
	tokenID.SetString(req.TokenID, 10)
	if tokenID.Sign() == 0 {
		tokenID.SetString(req.TokenID, 0)
	}

	expiration := big.NewInt(req.Expiration)

	// Build order payload
	order := OrderPayload{
		Salt:          RandomSalt(),
		Maker:         c.wallet,
		Signer:        c.signer.Address(),
		Taker:         common.Address{},
		TokenID:       tokenID,
		MakerAmount:   makerAmt,
		TakerAmount:   takerAmt,
		Expiration:    expiration,
		Nonce:         big.NewInt(0),
		FeeRateBps:    big.NewInt(int64(feeRate)),
		Side:          uint8(req.Side),
		SignatureType: SigTypeEOA,
	}

	// Sign the order
	signature, err := c.signer.SignOrder(order, negRisk)
	if err != nil {
		return ClobOrderResponse{}, fmt.Errorf("sign order: %w", err)
	}

	body := map[string]interface{}{
		"order": map[string]interface{}{
			"salt":          order.Salt.Int64(),
			"maker":         c.wallet.Hex(),
			"signer":        c.signer.Address().Hex(),
			"taker":         common.Address{}.Hex(),
			"tokenId":       req.TokenID,
			"makerAmount":   makerAmt.String(),
			"takerAmount":   takerAmt.String(),
			"expiration":    expiration.String(),
			"nonce":         "0",
			"feeRateBps":    strconv.Itoa(feeRate),
			"side":          sideToClob(req.Side),
			"signatureType": order.SignatureType,
			"signature":     signature,
		},
		"owner":     c.config.APIKey,
		"orderType": orderType,
	}

	if postOnly {
		body["postOnly"] = true
	}

	data, err := json.Marshal(body)
	if err != nil {
		return ClobOrderResponse{}, fmt.Errorf("marshal order: %w", err)
	}

	c.log.Debug().
		Str("token_id", req.TokenID).
		Str("side", sideToClob(req.Side)).
		Str("maker_amount", makerAmt.String()).
		Str("taker_amount", takerAmt.String()).
		Str("order_type", orderType).
		Bool("post_only", postOnly).
		Int("fee_rate_bps", feeRate).
		RawJSON("payload", data).
		Msg("submitting signed order")

	var resp ClobOrderResponse
	if err := c.doRequest("POST", "/order", data, &resp); err != nil {
		return resp, err
	}

	if !resp.Success {
		return resp, fmt.Errorf("clob order failed: %s", resp.ErrorMsg)
	}
	return resp, nil
}

// getFeeRate fetches the fee rate for a token from the CLOB API.
func (c *ClobClient) getFeeRate(tokenID string) (int, error) {
	var resp feeRateResp
	path := "/fee-rate?token_id=" + tokenID
	if err := c.doRequest("GET", path, nil, &resp); err != nil {
		return 0, err
	}
	return resp.BaseFee, nil
}

// getNegRisk checks if a token is on the neg-risk exchange.
func (c *ClobClient) getNegRisk(tokenID string) bool {
	var resp negRiskResp
	path := "/neg-risk?token_id=" + tokenID
	if err := c.doRequest("GET", path, nil, &resp); err != nil {
		c.log.Warn().Err(err).Msg("failed to fetch neg_risk, assuming false")
		return false
	}
	return resp.NegRisk
}

// getTickSize fetches and caches the tick size for a token from the CLOB API.
func (c *ClobClient) getTickSize(tokenID string) string {
	if ts, ok := c.tickSizeCache[tokenID]; ok {
		return ts
	}
	var resp tickSizeResp
	path := "/tick-size?token_id=" + tokenID
	if err := c.doRequest("GET", path, nil, &resp); err != nil {
		c.log.Warn().Err(err).Msg("failed to fetch tick size, using default 0.01")
		return "0.01"
	}
	ts := strconv.FormatFloat(resp.MinimumTickSize, 'f', -1, 64)
	c.tickSizeCache[tokenID] = ts
	c.log.Debug().Str("token_id", tokenID).Str("tick_size", ts).Msg("fetched tick size")
	return ts
}

// CancelOrder cancels a single order.
func (c *ClobClient) CancelOrder(orderID string) error {
	body, _ := json.Marshal(map[string]string{"orderID": orderID})
	return c.doRequest("DELETE", "/order", body, nil)
}

// CancelMarketOrders cancels all orders for a condition ID.
func (c *ClobClient) CancelMarketOrders(conditionID string) error {
	body, _ := json.Marshal(map[string]string{"market": conditionID})
	return c.doRequest("DELETE", "/cancel-market-orders", body, nil)
}

// CancelAll cancels all resting orders.
func (c *ClobClient) CancelAll() error {
	return c.doRequest("DELETE", "/cancel-all", nil, nil)
}

// GetOrderStatus fetches the current status of a single order from the CLOB.
func (c *ClobClient) GetOrderStatus(orderID string) (ClobOrderStatusResp, error) {
	var resp ClobOrderStatusResp
	path := "/data/order/" + orderID
	if err := c.doRequest("GET", path, nil, &resp); err != nil {
		return resp, err
	}
	return resp, nil
}

func (c *ClobClient) doRequest(method, path string, body []byte, result interface{}) error {
	url := c.config.BaseURL + path

	var bodyReader io.Reader
	if body != nil {
		bodyReader = bytes.NewReader(body)
	}

	req, err := http.NewRequest(method, url, bodyReader)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	c.signRequest(req, method, path, body)

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return fmt.Errorf("clob api error %d: %s", resp.StatusCode, string(respBody))
	}

	if result != nil {
		if err := json.Unmarshal(respBody, result); err != nil {
			return fmt.Errorf("unmarshal response: %w", err)
		}
	}
	return nil
}

// signRequest adds L2 HMAC-SHA256 auth headers.
func (c *ClobClient) signRequest(req *http.Request, method, path string, body []byte) {
	ts := time.Now().Unix()

	message := fmt.Sprintf("%d%s%s", ts, method, path)
	if len(body) > 0 {
		message += string(body)
	}

	key, _ := base64.URLEncoding.DecodeString(c.config.APISecret)
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(message))
	sig := base64.URLEncoding.EncodeToString(mac.Sum(nil))

	req.Header.Set("POLY_ADDRESS", c.config.WalletAddress)
	req.Header.Set("POLY_SIGNATURE", sig)
	req.Header.Set("POLY_TIMESTAMP", strconv.FormatInt(ts, 10))
	req.Header.Set("POLY_API_KEY", c.config.APIKey)
	req.Header.Set("POLY_PASSPHRASE", c.config.APIPassphrase)
}

func sideToClob(s Side) string {
	if s == Buy {
		return "BUY"
	}
	return "SELL"
}

func (s Side) opposite() Side {
	if s == Buy {
		return Sell
	}
	return Buy
}
