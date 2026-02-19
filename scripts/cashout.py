#!/usr/bin/env python3
"""Withdraw USDC.e from trading wallet to destination address on Polygon."""

import json
import os
import sys

from eth_account import Account
from web3 import Web3

USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
ERC20_ABI = json.loads('[{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"type":"function"}]')

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DEFAULT_DEST = os.getenv("CASHOUT_DEST", "")  # Set CASHOUT_DEST in .env
RPC_URL = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")


def main():

    private_key = os.getenv("PRIVATE_KEY")
    if not private_key:
        print("ERROR: PRIVATE_KEY not set in .env")
        sys.exit(1)

    dest = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DEST

    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    account = Account.from_key(private_key)
    usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC_E), abi=ERC20_ABI)

    balance = usdc.functions.balanceOf(account.address).call()
    balance_human = balance / 1e6

    print(f"Wallet:  {account.address}")
    print(f"Balance: ${balance_human:.2f} USDC.e")
    print(f"Dest:    {dest}")

    if balance == 0:
        print("Nothing to withdraw.")
        return

    # Allow partial withdrawal via second arg
    if len(sys.argv) > 2:
        amount = int(float(sys.argv[2]) * 1e6)
        if amount > balance:
            print(f"ERROR: requested ${float(sys.argv[2]):.2f} but only have ${balance_human:.2f}")
            sys.exit(1)
    else:
        amount = balance

    amount_human = amount / 1e6
    confirm = input(f"\nSend ${amount_human:.2f} USDC.e to {dest}? [y/N] ")
    if confirm.lower() != "y":
        print("Aborted.")
        return

    nonce = w3.eth.get_transaction_count(account.address)
    tx = usdc.functions.transfer(
        Web3.to_checksum_address(dest), amount
    ).build_transaction({
        "from": account.address,
        "nonce": nonce,
        "gas": 100_000,
        "gasPrice": w3.eth.gas_price,
        "chainId": 137,
    })

    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"Tx sent: https://polygonscan.com/tx/{tx_hash.hex()}")

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt.status == 1:
        print(f"Success! ${amount_human:.2f} USDC.e sent to {dest}")
    else:
        print("Transaction failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
