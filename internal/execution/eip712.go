package execution

import (
	"crypto/ecdsa"
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

// Contract addresses per chain.
var (
	ctfExchange = map[int]common.Address{
		137:   common.HexToAddress("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"), // Polygon mainnet
		80002: common.HexToAddress("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"), // Amoy testnet
	}
	negRiskCTFExchange = map[int]common.Address{
		137:   common.HexToAddress("0xC5d563A36AE78145C45a50134d48A1215220f80a"), // Polygon mainnet
		80002: common.HexToAddress("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"), // Amoy testnet
	}
)

// EIP-712 type hashes (precomputed).
var (
	eip712DomainTypeHash = crypto.Keccak256Hash([]byte(
		"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
	))
	orderTypeHash = crypto.Keccak256Hash([]byte(
		"Order(uint256 salt,address maker,address signer,address taker,uint256 tokenId,uint256 makerAmount,uint256 takerAmount,uint256 expiration,uint256 nonce,uint256 feeRateBps,uint8 side,uint8 signatureType)",
	))
)

// SignatureType for the order.
const (
	SigTypeEOA       uint8 = 0
	SigTypePolyProxy uint8 = 1
)

// OrderPayload is the full order struct for EIP-712 signing.
type OrderPayload struct {
	Salt          *big.Int
	Maker         common.Address
	Signer        common.Address
	Taker         common.Address // 0x0 for open orders
	TokenID       *big.Int
	MakerAmount   *big.Int
	TakerAmount   *big.Int
	Expiration    *big.Int
	Nonce         *big.Int
	FeeRateBps    *big.Int
	Side          uint8 // 0=BUY, 1=SELL
	SignatureType uint8
}

// OrderSigner handles EIP-712 signing for Polymarket CTFExchange orders.
type OrderSigner struct {
	key            *ecdsa.PrivateKey
	chainID        int
	domainSepCache map[common.Address]common.Hash
}

// NewOrderSigner creates a signer for the given chain.
func NewOrderSigner(key *ecdsa.PrivateKey, chainID int) (*OrderSigner, error) {
	// Verify we have exchange addresses for this chain
	if _, ok := ctfExchange[chainID]; !ok {
		return nil, fmt.Errorf("unsupported chain ID %d", chainID)
	}

	return &OrderSigner{
		key:            key,
		chainID:        chainID,
		domainSepCache: make(map[common.Address]common.Hash),
	}, nil
}

func (s *OrderSigner) domainSeparator(exchange common.Address) common.Hash {
	if ds, ok := s.domainSepCache[exchange]; ok {
		return ds
	}
	ds := crypto.Keccak256Hash(
		eip712DomainTypeHash.Bytes(),
		crypto.Keccak256Hash([]byte("Polymarket CTF Exchange")).Bytes(),
		crypto.Keccak256Hash([]byte("1")).Bytes(),
		common.LeftPadBytes(big.NewInt(int64(s.chainID)).Bytes(), 32),
		common.LeftPadBytes(exchange.Bytes(), 32),
	)
	s.domainSepCache[exchange] = ds
	return ds
}

// SignOrder signs the order payload and returns the hex-encoded signature.
func (s *OrderSigner) SignOrder(order OrderPayload, negRisk bool) (string, error) {
	// Select exchange based on neg_risk
	exchanges := ctfExchange
	if negRisk {
		exchanges = negRiskCTFExchange
	}
	exchange := exchanges[s.chainID]
	domSep := s.domainSeparator(exchange)

	structHash := hashOrder(order)

	// EIP-712: "\x19\x01" + domainSeparator + structHash
	payload := make([]byte, 2+32+32)
	payload[0] = 0x19
	payload[1] = 0x01
	copy(payload[2:34], domSep.Bytes())
	copy(payload[34:66], structHash.Bytes())

	digest := crypto.Keccak256Hash(payload)

	sig, err := crypto.Sign(digest.Bytes(), s.key)
	if err != nil {
		return "", fmt.Errorf("sign: %w", err)
	}

	// Ethereum signature: adjust v from 0/1 to 27/28
	sig[64] += 27

	return fmt.Sprintf("0x%x", sig), nil
}

// Address returns the signer's Ethereum address.
func (s *OrderSigner) Address() common.Address {
	return crypto.PubkeyToAddress(s.key.PublicKey)
}

func hashOrder(o OrderPayload) common.Hash {
	return crypto.Keccak256Hash(
		orderTypeHash.Bytes(),
		pad32(o.Salt),
		common.LeftPadBytes(o.Maker.Bytes(), 32),
		common.LeftPadBytes(o.Signer.Bytes(), 32),
		common.LeftPadBytes(o.Taker.Bytes(), 32),
		pad32(o.TokenID),
		pad32(o.MakerAmount),
		pad32(o.TakerAmount),
		pad32(o.Expiration),
		pad32(o.Nonce),
		pad32(o.FeeRateBps),
		pad32(big.NewInt(int64(o.Side))),
		pad32(big.NewInt(int64(o.SignatureType))),
	)
}

func pad32(v *big.Int) []byte {
	if v == nil {
		return make([]byte, 32)
	}
	return common.LeftPadBytes(v.Bytes(), 32)
}

// RandomSalt generates a random 32-bit salt (matching py-clob-client's generate_seed).
func RandomSalt() *big.Int {
	max := new(big.Int).SetInt64(1 << 31) // 2^31
	salt, _ := rand.Int(rand.Reader, max)
	return salt
}

// RoundConfig defines decimal precision for a given tick size, matching py-clob-client.
type RoundConfig struct {
	Price  int // decimal places for price rounding
	Size   int // decimal places for size rounding (always 2)
	Amount int // max decimal places for the USDC amount (product)
}

// roundingConfigs maps tick size string → rounding precision.
var roundingConfigs = map[string]RoundConfig{
	"0.1":    {Price: 1, Size: 2, Amount: 3},
	"0.01":   {Price: 2, Size: 2, Amount: 4},
	"0.001":  {Price: 3, Size: 2, Amount: 5},
	"0.0001": {Price: 4, Size: 2, Amount: 6},
}

// GetRoundConfig returns the rounding config for a tick size string.
// Defaults to 0.01 config if tick size is unknown.
func GetRoundConfig(tickSize string) RoundConfig {
	if rc, ok := roundingConfigs[tickSize]; ok {
		return rc
	}
	return roundingConfigs["0.01"]
}

// FOK precision limits (stricter than GTC, applies to all tick sizes).
// See: https://github.com/Polymarket/py-clob-client/issues/121
const (
	fokMakerMaxDec = 2 // FOK makerAmount max decimal places
	fokTakerMaxDec = 4 // FOK takerAmount max decimal places
)

// ComputeAmounts calculates maker/taker amounts from price and size.
// Amounts are in 6-decimal fixed-point (USDC uses 6 decimals).
// Applies rounding matching py-clob-client: price rounded to tick precision,
// size floored to 2 dec, product adaptively rounded to amount precision.
// For FOK orders, stricter precision limits apply (max 2 dec for USDC amount).
func ComputeAmounts(price, size float64, side Side, tickSize string, isFOK bool) (makerAmount, takerAmount *big.Int) {
	rc := GetRoundConfig(tickSize)

	// Round price to tick precision (normal rounding)
	price = roundNormal(price, rc.Price)

	// Floor size to 2 decimal places
	size = roundDown(size, rc.Size)

	// Determine max amount decimal places
	maxAmountDec := rc.Amount
	if isFOK && maxAmountDec > fokMakerMaxDec {
		maxAmountDec = fokMakerMaxDec
	}

	// For FOK: adjust size down so price*size has ≤ maxAmountDec decimal places
	if isFOK {
		size = adjustSizeForPrecision(price, size, rc.Price, maxAmountDec)
	}

	// Compute product
	usdc := price * size

	// Adaptive rounding on the product amount
	usdc = roundAmount(usdc, maxAmountDec)

	switch side {
	case Buy:
		// BUY: maker pays USDC, taker receives shares
		makerAmount = toTokenDecimals(usdc)
		takerAmount = toTokenDecimals(size)
	default:
		// SELL: maker gives shares, taker receives USDC
		makerAmount = toTokenDecimals(size)
		takerAmount = toTokenDecimals(usdc)
	}
	return
}

// adjustSizeForPrecision floors size so that price × size has at most maxDec
// decimal places. Works in integer arithmetic to avoid float imprecision.
func adjustSizeForPrecision(price, size float64, priceDec, maxDec int) float64 {
	const sizeDec = 2
	exponent := priceDec + sizeDec - maxDec
	if exponent <= 0 {
		return size // product already fits
	}

	P := int64(math.Round(price * math.Pow10(priceDec)))
	S := int64(math.Floor(size * math.Pow10(sizeDec)))

	// P * S must be divisible by 10^exponent
	divisor := int64(math.Pow10(exponent))
	g := gcd64(P, divisor)
	step := divisor / g

	// Floor S to nearest multiple of step
	S = (S / step) * step
	if S <= 0 {
		S = step // minimum valid size
	}

	return float64(S) / math.Pow10(sizeDec)
}

func gcd64(a, b int64) int64 {
	if a < 0 {
		a = -a
	}
	if b < 0 {
		b = -b
	}
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// roundAmount applies py-clob-client's adaptive rounding:
// 1. If decimal places are within limit, keep as-is
// 2. Try rounding up with extra precision (catches FP artifacts like 2.49999→2.5)
// 3. Fall back to floor at the limit
func roundAmount(x float64, maxDecimals int) float64 {
	if decimalPlaces(x) <= maxDecimals {
		return x
	}
	// Try rounding up with extra precision
	rounded := roundUp(x, maxDecimals+4)
	if decimalPlaces(rounded) <= maxDecimals {
		return rounded
	}
	// Fall back to floor
	return roundDown(x, maxDecimals)
}

// toTokenDecimals converts a float to 6-decimal fixed-point integer (matching USDC).
func toTokenDecimals(x float64) *big.Int {
	f := x * 1e6
	if decimalPlaces(f) > 0 {
		f = math.Round(f)
	}
	return big.NewInt(int64(f))
}

func roundNormal(x float64, digits int) float64 {
	scale := math.Pow10(digits)
	return math.Round(x*scale) / scale
}

func roundDown(x float64, digits int) float64 {
	scale := math.Pow10(digits)
	return math.Floor(x*scale) / scale
}

func roundUp(x float64, digits int) float64 {
	scale := math.Pow10(digits)
	return math.Ceil(x*scale) / scale
}

// decimalPlaces counts decimal places in a float, matching py-clob-client's Decimal approach.
func decimalPlaces(x float64) int {
	s := strconv.FormatFloat(x, 'f', -1, 64)
	dot := strings.IndexByte(s, '.')
	if dot < 0 {
		return 0
	}
	return len(s) - dot - 1
}
