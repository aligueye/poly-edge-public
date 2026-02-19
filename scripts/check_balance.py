#!/usr/bin/env python3
"""Check trading wallet balances and CTF Exchange allowances on Polygon."""

import os
import sys
import time

from dotenv import load_dotenv
from web3 import Web3

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

RPC_URL = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
NATIVE_USDC = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
CTF_EXCHANGE = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_CTF = "0xC5d563A36AE78145C45a50134d48A1215220f80a"

ERC20_ABI = [
    {"constant": True, "inputs": [{"name": "owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
    {"constant": True, "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}], "name": "allowance", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "symbol", "outputs": [{"name": "", "type": "string"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
]

w3 = Web3(Web3.HTTPProvider(RPC_URL))

wallet = os.getenv("WALLET_ADDRESS")
if not wallet:
    key = os.getenv("POLY_PRIVATE_KEY", "")
    if key.startswith("0x"):
        key = key[2:]
    if len(key) == 64:
        from eth_account import Account
        wallet = Account.from_key(key).address
    else:
        print("Set WALLET_ADDRESS or POLY_PRIVATE_KEY in .env")
        sys.exit(1)

bankroll = float(os.getenv("BANKROLL_USDC", "0"))

print(f"Wallet:   {wallet}")
print(f"Bankroll: ${bankroll:.2f}")
print()

# POL balance
pol = w3.eth.get_balance(wallet)
print(f"POL (gas):     {pol / 1e18:.4f}")

def fmt_allowance(a, d):
    v = a / (10 ** d)
    return "unlimited" if v > 1e30 else f"${v:.2f}"

def rpc_call(fn):
    """Retry with backoff on rate limit."""
    for attempt in range(5):
        try:
            return fn()
        except Exception as e:
            if "rate limit" in str(e).lower() or "Too many" in str(e):
                time.sleep(4 * (attempt + 1))
            else:
                raise
    return fn()

# Token balances and allowances
for label, addr in [("USDC.e", USDC_E), ("USDC (native)", NATIVE_USDC)]:
    token = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=ERC20_ABI)
    bal = rpc_call(lambda: token.functions.balanceOf(wallet).call())
    time.sleep(1)
    decimals = rpc_call(lambda: token.functions.decimals().call())
    bal_human = bal / (10 ** decimals)
    time.sleep(1)

    ctf_allowance = rpc_call(lambda: token.functions.allowance(wallet, CTF_EXCHANGE).call())
    time.sleep(1)
    neg_allowance = rpc_call(lambda: token.functions.allowance(wallet, NEG_RISK_CTF).call())
    time.sleep(1)

    print(f"{label:14s} ${bal_human:.6f}   CTF allowance: {fmt_allowance(ctf_allowance, decimals)}   NegRisk allowance: {fmt_allowance(neg_allowance, decimals)}")

print()

# Summary — reuse data already fetched above
time.sleep(1)
usdc_e = w3.eth.contract(address=Web3.to_checksum_address(USDC_E), abi=ERC20_ABI)
bal = rpc_call(lambda: usdc_e.functions.balanceOf(wallet).call()) / 1e6
time.sleep(1)
ctf_ok = rpc_call(lambda: usdc_e.functions.allowance(wallet, CTF_EXCHANGE).call()) / 1e6 > 1e30

if bal < 1:
    print(f"!! Wallet has ${bal:.2f} USDC.e — bot cannot trade")
    print(f"   Send USDC.e (Polygon) to {wallet}")
elif bal < bankroll:
    print(f"!! Wallet has ${bal:.2f} but bankroll is ${bankroll:.2f}")
    print(f"   Fund ${bankroll - bal:.2f} more or lower BANKROLL_USDC in .env")
else:
    print(f"OK — ${bal:.2f} USDC.e available (bankroll ${bankroll:.2f})")

if not ctf_ok:
    print("!! CTF Exchange allowance not set — need to approve USDC.e spending")
