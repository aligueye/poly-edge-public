"""gRPC sidecar server — serves ML predictions to the Go process.

Listens on a Unix Domain Socket. Receives MarketState from Go,
computes features, runs model inference, returns Signal.

Usage:
    .venv/bin/python python/sidecar/server.py [--socket /tmp/polyedge.sock] [--model-dir data/models/]
"""

import argparse
import logging
import os
import sys
from concurrent import futures

# Add proto dir to path so generated stubs can import each other
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "proto"))

import grpc
import signals_pb2
import signals_pb2_grpc

from features import compute_features
from model import Model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("sidecar")


class SignalServicer(signals_pb2_grpc.SignalServiceServicer):
    """Implements the SignalService gRPC interface."""

    def __init__(self, model: Model):
        self._model = model
        self._call_count = 0

    def GetSignal(self, request, context):
        self._call_count += 1

        # Compute features from MarketState
        features = compute_features(request)

        # Run model inference
        action, confidence = self._model.predict(features)

        if self._call_count <= 5 or self._call_count % 100 == 0:
            logger.info(
                "call #%d: action=%s confidence=%.4f btc=%.2f yes=%.4f",
                self._call_count,
                action,
                confidence,
                request.btc_price,
                request.polymarket_yes_price,
            )

        return signals_pb2.Signal(
            action=action,
            confidence=confidence,
            suggested_price=request.polymarket_yes_price,
            suggested_size=0.0,
        )


def serve(socket_path: str, model_dir: str):
    """Start the gRPC server on a Unix Domain Socket."""
    # Clean up stale socket
    if os.path.exists(socket_path):
        os.remove(socket_path)

    model = Model(model_dir)
    if not model.loaded:
        logger.warning("no trained model — all predictions will be 'hold'")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    signals_pb2_grpc.add_SignalServiceServicer_to_server(
        SignalServicer(model), server
    )
    server.add_insecure_port(f"unix:{socket_path}")
    server.start()

    logger.info("sidecar listening on unix:%s", socket_path)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("shutting down...")
        server.stop(grace=5)


def main():
    parser = argparse.ArgumentParser(description="poly-edge ML sidecar")
    parser.add_argument(
        "--socket",
        default=os.environ.get("GRPC_SOCKET_PATH", "/tmp/polyedge.sock"),
        help="UDS path (default: /tmp/polyedge.sock)",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "data/models"),
        help="Model directory with training_meta.json (default: data/models/)",
    )
    args = parser.parse_args()
    serve(args.socket, args.model_dir)


if __name__ == "__main__":
    main()
