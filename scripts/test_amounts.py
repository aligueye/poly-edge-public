"""Test py-clob-client amount computation for various price/size combos."""
import sys
sys.path.insert(0, "/home/ali/dev/poly-edge/.venv/lib/python3.12/site-packages")

from py_clob_client.order_builder.helpers import round_down, round_normal, round_up, to_token_decimals, decimal_places
from py_clob_client.clob_types import RoundConfig

# Tick size 0.01 config
rc = RoundConfig(price=2, size=2, amount=4)

test_cases = [
    (0.37, 6.66, "BUY"),   # current failing
    (0.495, 5.05, "BUY"),  # typical
    (0.49, 5.05, "BUY"),   # previous success
    (0.479, 5.2, "BUY"),   # from error msg
    (0.489, 5.05, "BUY"),  # from error msg
    (0.488, 3.32, "BUY"),  # from error msg
]

for price, size, side in test_cases:
    raw_price = round_normal(price, rc.price)
    raw_taker = round_down(size, rc.size)
    raw_maker = raw_taker * raw_price

    print(f"\nprice={price}, size={size}, side={side}")
    print(f"  raw_price={raw_price}, raw_taker={raw_taker}, raw_maker={raw_maker}")
    print(f"  decimal_places(raw_maker)={decimal_places(raw_maker)}")

    if decimal_places(raw_maker) > rc.amount:
        raw_maker_r = round_up(raw_maker, rc.amount + 4)
        print(f"  after round_up({rc.amount+4}): {raw_maker_r}, dec={decimal_places(raw_maker_r)}")
        if decimal_places(raw_maker_r) > rc.amount:
            raw_maker_r = round_down(raw_maker_r, rc.amount)
            print(f"  after round_down({rc.amount}): {raw_maker_r}")
        raw_maker = raw_maker_r

    maker_amount = to_token_decimals(raw_maker)
    taker_amount = to_token_decimals(raw_taker)

    print(f"  RESULT: makerAmount={maker_amount}, takerAmount={taker_amount}")
    print(f"  maker in USD: {maker_amount/1e6}, taker in tokens: {taker_amount/1e6}")
    print(f"  derived price: {maker_amount/taker_amount:.6f}")
