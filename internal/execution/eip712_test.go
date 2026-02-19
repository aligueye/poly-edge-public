package execution

import (
	"fmt"
	"math/big"
	"testing"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/crypto"
)

// TestEIP712IntermediateValues computes the same EIP-712 intermediate values
// as scripts/eip712_test.py and compares them to find divergences.
func TestEIP712IntermediateValues(t *testing.T) {
	// Deterministic test parameters (same as Python script)
	salt := big.NewInt(12345678)
	maker := common.HexToAddress("0x65C6ABD9AF31B7239782a91d571Edc8BF532fd7f")
	signer := common.HexToAddress("0x65C6ABD9AF31B7239782a91d571Edc8BF532fd7f")
	taker := common.HexToAddress("0x0000000000000000000000000000000000000000")
	tokenID := new(big.Int)
	tokenID.SetString("77760253819246575134313219259732112720116263140867452724940244809905063878295", 10)
	makerAmount := big.NewInt(1943681)
	takerAmount := big.NewInt(4740686)
	expiration := big.NewInt(0)
	nonce := big.NewInt(0)
	feeRateBps := big.NewInt(1000)
	side := uint8(0)          // BUY
	signatureType := uint8(1) // POLY_PROXY

	chainID := 137
	exchange := common.HexToAddress("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E")

	// Expected values from Python script
	expectedDomainTypeHash := "0x8b73c3c69bb8fe3d512ecc4cf759cc79239f7b179b0ffacaa9a75d522b39400f"
	expectedDomainSep := "0x1a573e3617c78403b5b4b892827992f027b03d4eaf570048b8ee8cdd84d151be"
	expectedOrderTypeHash := "0xa852566c4e14d00869b6db0220888a9090a13eccdaea03713ff0a3d27bf9767c"
	expectedStructHash := "0x165946a607547cfaa09c63244aa18d54aa94dced67560e8e29bbd04289720125"
	expectedDigest := "0xf96a7b003538d833ad6631c90730cc2fac7eb88e3a394ea3d0cdca09db2597b2"

	// 1. Domain type hash
	domainTypeHash := crypto.Keccak256Hash([]byte(
		"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)",
	))
	got := fmt.Sprintf("0x%x", domainTypeHash)
	t.Logf("Domain type hash:    %s", got)
	if got != expectedDomainTypeHash {
		t.Errorf("Domain type hash mismatch!\n  got:  %s\n  want: %s", got, expectedDomainTypeHash)
	}

	// 2. Domain separator
	domainSep := crypto.Keccak256Hash(
		domainTypeHash.Bytes(),
		crypto.Keccak256Hash([]byte("Polymarket CTF Exchange")).Bytes(),
		crypto.Keccak256Hash([]byte("1")).Bytes(),
		common.LeftPadBytes(big.NewInt(int64(chainID)).Bytes(), 32),
		common.LeftPadBytes(exchange.Bytes(), 32),
	)
	got = fmt.Sprintf("0x%x", domainSep)
	t.Logf("Domain separator:    %s", got)
	if got != expectedDomainSep {
		t.Errorf("Domain separator mismatch!\n  got:  %s\n  want: %s", got, expectedDomainSep)
	}

	// 3. Order type hash
	got = fmt.Sprintf("0x%x", orderTypeHash)
	t.Logf("Order type hash:     %s", got)
	if got != expectedOrderTypeHash {
		t.Errorf("Order type hash mismatch!\n  got:  %s\n  want: %s", got, expectedOrderTypeHash)
	}

	// 4. Struct hash
	order := OrderPayload{
		Salt:          salt,
		Maker:         maker,
		Signer:        signer,
		Taker:         taker,
		TokenID:       tokenID,
		MakerAmount:   makerAmount,
		TakerAmount:   takerAmount,
		Expiration:    expiration,
		Nonce:         nonce,
		FeeRateBps:    feeRateBps,
		Side:          side,
		SignatureType: signatureType,
	}
	structHash := hashOrder(order)
	got = fmt.Sprintf("0x%x", structHash)
	t.Logf("Struct hash (order): %s", got)
	if got != expectedStructHash {
		t.Errorf("Struct hash mismatch!\n  got:  %s\n  want: %s", got, expectedStructHash)
	}

	// 5. Final digest
	payload := make([]byte, 2+32+32)
	payload[0] = 0x19
	payload[1] = 0x01
	copy(payload[2:34], domainSep.Bytes())
	copy(payload[34:66], structHash.Bytes())
	digest := crypto.Keccak256Hash(payload)
	got = fmt.Sprintf("0x%x", digest)
	t.Logf("Final digest:        %s", got)
	if got != expectedDigest {
		t.Errorf("Final digest mismatch!\n  got:  %s\n  want: %s", got, expectedDigest)
	}

	// Also print individual field encodings for debugging
	t.Log("\nField encodings:")
	t.Logf("  salt:          %x", pad32(salt))
	t.Logf("  maker:         %x", common.LeftPadBytes(maker.Bytes(), 32))
	t.Logf("  signer:        %x", common.LeftPadBytes(signer.Bytes(), 32))
	t.Logf("  taker:         %x", common.LeftPadBytes(taker.Bytes(), 32))
	t.Logf("  tokenId:       %x", pad32(tokenID))
	t.Logf("  makerAmount:   %x", pad32(makerAmount))
	t.Logf("  takerAmount:   %x", pad32(takerAmount))
	t.Logf("  expiration:    %x", pad32(expiration))
	t.Logf("  nonce:         %x", pad32(nonce))
	t.Logf("  feeRateBps:    %x", pad32(feeRateBps))
	t.Logf("  side:          %x", pad32(big.NewInt(int64(side))))
	t.Logf("  signatureType: %x", pad32(big.NewInt(int64(signatureType))))
}
