"""
Compute EIP-712 intermediate values for a deterministic Polymarket order.

Uses py_order_utils + poly_eip712_structs to construct an Order struct and
print all intermediate hashes needed to verify an EIP-712 signing implementation.

Run with:
    .venv/bin/python scripts/eip712_test.py
"""

from eth_utils.crypto import keccak
from poly_eip712_structs import make_domain
from py_order_utils.model.order import Order
from py_order_utils.utils import normalize_address

# ---------------------------------------------------------------------------
# 1. Deterministic test parameters
# ---------------------------------------------------------------------------
SALT = 12345678
MAKER = "0x65C6ABD9AF31B7239782a91d571Edc8BF532fd7f"
SIGNER = "0x65C6ABD9AF31B7239782a91d571Edc8BF532fd7f"
TAKER = "0x0000000000000000000000000000000000000000"
TOKEN_ID = "77760253819246575134313219259732112720116263140867452724940244809905063878295"
MAKER_AMOUNT = "1943681"
TAKER_AMOUNT = "4740686"
EXPIRATION = "0"
NONCE = "0"
FEE_RATE_BPS = "1000"
SIDE = 0         # BUY
SIGNATURE_TYPE = 1  # POLY_PROXY
CHAIN_ID = 137
EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # CTFExchange

# ---------------------------------------------------------------------------
# 2. Build the EIP-712 domain (matches BaseBuilder._get_domain_separator)
# ---------------------------------------------------------------------------
domain = make_domain(
    name="Polymarket CTF Exchange",
    version="1",
    chainId=str(CHAIN_ID),
    verifyingContract=normalize_address(EXCHANGE_ADDRESS),
)

# ---------------------------------------------------------------------------
# 3. Build the Order struct
# ---------------------------------------------------------------------------
order = Order(
    salt=SALT,
    maker=normalize_address(MAKER),
    signer=normalize_address(SIGNER),
    taker=normalize_address(TAKER),
    tokenId=int(TOKEN_ID),
    makerAmount=int(MAKER_AMOUNT),
    takerAmount=int(TAKER_AMOUNT),
    expiration=int(EXPIRATION),
    nonce=int(NONCE),
    feeRateBps=int(FEE_RATE_BPS),
    side=SIDE,
    signatureType=SIGNATURE_TYPE,
)

# ---------------------------------------------------------------------------
# 4. Compute intermediate EIP-712 values
# ---------------------------------------------------------------------------

# Domain type hash: keccak256 of the domain's encoded type string
# e.g. "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
domain_class = type(domain)
domain_type_hash = domain_class.type_hash()

# Domain separator: hash_struct of the domain
# hash_struct = keccak256(typeHash || encodeData)
domain_separator = domain.hash_struct()

# Order type hash: keccak256 of Order's encoded type string
order_type_hash = Order.type_hash()

# Struct hash (order hash): hash_struct of the order
# hash_struct = keccak256(typeHash || encodeData)
struct_hash = order.hash_struct()

# Final EIP-712 digest: keccak256(0x19 0x01 || domainSeparator || structHash)
signable = order.signable_bytes(domain=domain)
final_digest = keccak(signable)

# ---------------------------------------------------------------------------
# 5. Print results
# ---------------------------------------------------------------------------
print("=" * 72)
print("EIP-712 Intermediate Values for Polymarket Order")
print("=" * 72)

print()
print(f"Domain encode_type:\n  {domain_class.encode_type()}")
print(f"Order  encode_type:\n  {Order.encode_type()}")

print()
print(f"Domain type hash:      0x{domain_type_hash.hex()}")
print(f"Domain separator:      0x{domain_separator.hex()}")
print(f"Order type hash:       0x{order_type_hash.hex()}")
print(f"Struct hash (order):   0x{struct_hash.hex()}")
print(f"Final digest:          0x{final_digest.hex()}")

print()
print("=" * 72)
print("Order field values (as passed to struct)")
print("=" * 72)
for name, _ in Order.get_members():
    print(f"  {name:20s} = {order[name]}")
