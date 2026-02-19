#!/usr/bin/env python3
"""Cancel all open orders on the Polymarket CLOB."""

import base64
import hashlib
import hmac
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

CLOB_BASE = "https://clob.polymarket.com"


def build_headers(method, path, body=None):
    api_key = os.getenv("POLY_API_KEY", "")
    api_secret = os.getenv("POLY_API_SECRET", "")
    passphrase = os.getenv("POLY_API_PASSPHRASE", "")
    wallet = os.getenv("WALLET_ADDRESS", "")

    ts = str(int(time.time()))
    message = ts + method + path
    if body:
        message += body

    key = base64.urlsafe_b64decode(api_secret)
    sig = base64.urlsafe_b64encode(
        hmac.new(key, message.encode(), hashlib.sha256).digest()
    ).decode()

    return {
        "Content-Type": "application/json",
        "POLY_ADDRESS": wallet,
        "POLY_SIGNATURE": sig,
        "POLY_TIMESTAMP": ts,
        "POLY_API_KEY": api_key,
        "POLY_PASSPHRASE": passphrase,
    }


def get_open_orders():
    path = "/data/orders?state=live"
    headers = build_headers("GET", path)
    resp = requests.get(CLOB_BASE + path, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def cancel_all():
    path = "/cancel-all"
    headers = build_headers("DELETE", path)
    resp = requests.delete(CLOB_BASE + path, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def main():
    wallet = os.getenv("WALLET_ADDRESS", "")
    if not wallet:
        print("Set WALLET_ADDRESS in .env")
        sys.exit(1)

    if not os.getenv("POLY_API_KEY"):
        print("Set POLY_API_KEY, POLY_API_SECRET, POLY_API_PASSPHRASE in .env")
        sys.exit(1)

    proxy_host = os.getenv("SOCKS5_PROXY_HOST", "")
    if proxy_host:
        proxy_port = os.getenv("SOCKS5_PROXY_PORT", "1080")
        proxy_user = os.getenv("SOCKS5_PROXY_USER", "")
        proxy_pass = os.getenv("SOCKS5_PROXY_PASS", "")
        if proxy_user:
            proxy_url = f"socks5://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
        else:
            proxy_url = f"socks5://{proxy_host}:{proxy_port}"
        os.environ["ALL_PROXY"] = proxy_url
        print(f"Using proxy: {proxy_host}:{proxy_port}")

    print(f"Wallet: {wallet}\n")

    # Fetch open orders
    print("Fetching open orders...")
    try:
        orders = get_open_orders()
    except Exception as e:
        print(f"Failed to fetch orders: {e}")
        # Try cancel-all anyway
        orders = []

    if isinstance(orders, list) and len(orders) > 0:
        total_locked = 0.0
        print(f"Found {len(orders)} open order(s):\n")
        for o in orders:
            oid = o.get("id", o.get("orderID", "?"))[:16]
            side = o.get("side", "?")
            price = o.get("price", o.get("original_size", "?"))
            size = o.get("size_matched", o.get("original_size", "?"))
            maker = int(o.get("maker_amount", "0")) / 1e6
            total_locked += maker
            print(f"  {oid}...  {side:4s}  maker=${maker:.2f}")
        print(f"\nTotal locked: ~${total_locked:.2f}")
    else:
        print("No open orders found (or could not fetch).")

    print("\nCancelling all orders...")
    try:
        result = cancel_all()
        canceled = result.get("canceled", [])
        not_canceled = result.get("not_canceled", {})
        print(f"Canceled: {len(canceled)}")
        if not_canceled:
            print(f"Not canceled: {not_canceled}")
        if canceled:
            print("Done — USDC.e collateral should be released shortly.")
        else:
            print("No orders to cancel.")
    except Exception as e:
        print(f"Cancel failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
