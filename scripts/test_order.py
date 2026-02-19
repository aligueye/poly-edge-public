"""Place a test order using py-clob-client to see exact HTTP request."""
import os, json
from dotenv import load_dotenv
load_dotenv("/home/ali/dev/poly-edge/.env")

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType

host = "https://clob.polymarket.com"
key = os.getenv("PRIVATE_KEY")
if key and key.startswith("0x"):
    key = key[2:]
chain_id = 137

client = ClobClient(
    host,
    key=key,
    chain_id=chain_id,
    creds={
        "apiKey": os.getenv("POLY_API_KEY"),
        "secret": os.getenv("POLY_API_SECRET"),
        "passphrase": os.getenv("POLY_API_PASSPHRASE"),
    },
)

token_id = "29764063228367759493800459940056768399426699679782711032229390569629618918707"

tick_size = client.get_tick_size(token_id)
neg_risk = client.get_neg_risk(token_id)
print(f"Tick size: {tick_size}")
print(f"Neg risk: {neg_risk}")

# Build order using proper OrderArgs
order_args = OrderArgs(
    token_id=token_id,
    price=0.37,
    size=6.66,
)
order = client.create_order(order_args)

print("\nOrder data:")
print(json.dumps(order, indent=2, default=str))

# Also show the amounts from the builder directly
from py_clob_client.order_builder.constants import BUY
builder = client.builder
side, maker_amount, taker_amount = builder.get_order_amounts(
    BUY, 6.66, 0.37, builder.get_round_config(token_id, tick_size)
)
print(f"\nDirect builder amounts: maker={maker_amount}, taker={taker_amount}")
print(f"  maker USD: {maker_amount/1e6}, taker tokens: {taker_amount/1e6}")
