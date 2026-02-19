.PHONY: build run test proto backfill train backtest docker-up docker-down cashout collect balance cancel

build:
	go build -o poly-edge ./cmd/poly-edge

test:
	go test ./...

proto:
	protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative internal/grpc/proto/signals.proto
	.venv/bin/python -m grpc_tools.protoc -Iinternal/grpc/proto \
		--python_out=python/sidecar/proto \
		--grpc_python_out=python/sidecar/proto \
		internal/grpc/proto/signals.proto

backfill:
	.venv/bin/python python/research/backfill.py --days 30 --output-dir data/training/

train:
	.venv/bin/python python/research/train_model.py --data-dir data/training/

backtest:
	.venv/bin/python python/research/backtest.py --data-dir data/training/

# Run: make run live | make run paper
# Starts sidecar + Go binary, cleans up sidecar on exit
ifeq ($(word 2,$(MAKECMDGOALS)),paper)
  RUN_ENV := PAPER_TRADE=true
  RUN_LOG := paper
else
  RUN_ENV :=
  RUN_LOG := live
endif

run:
	@mkdir -p logs
	@go build -o poly-edge ./cmd/poly-edge
	@.venv/bin/python python/sidecar/server.py & SIDECAR_PID=$$!; \
	trap "kill $$SIDECAR_PID 2>/dev/null" EXIT; \
	sleep 1; \
	$(RUN_ENV) ./poly-edge 2>&1 | (trap '' INT; tee logs/$(RUN_LOG)-$$(date +%Y%m%d-%H%M%S).log) || true

balance:
	.venv/bin/python scripts/check_balance.py

cancel:
	.venv/bin/python scripts/cancel_orders.py

cashout:
	.venv/bin/python scripts/cashout.py

# Collector: make collect [start|stop|restart|status|logs]
collect:
ifeq ($(word 2,$(MAKECMDGOALS)),stop)
	@systemctl --user stop poly-collector.service && echo "Collector stopped"
else ifeq ($(word 2,$(MAKECMDGOALS)),restart)
	@systemctl --user restart poly-collector.service && echo "Collector restarted"
else ifeq ($(word 2,$(MAKECMDGOALS)),status)
	@systemctl --user is-active poly-collector.service --quiet \
		&& echo "Collector running (PID $$(systemctl --user show poly-collector.service -p MainPID --value))" \
		&& echo "Since: $$(systemctl --user show poly-collector.service -p ActiveEnterTimestamp --value)" \
		&& ls -lh data/live/live_*.parquet 2>/dev/null | tail -1 \
		|| echo "Collector not running"
else ifeq ($(word 2,$(MAKECMDGOALS)),logs)
	@tail -20 data/live/collector.log
else
	@systemctl --user start poly-collector.service && echo "Collector started"
endif

# Catch subcommands so make doesn't error
.PHONY: stop start restart status logs live paper
stop start restart status logs live paper:
	@true

docker-up:
	docker compose -f deploy/docker-compose.yml up --build -d

docker-down:
	docker compose -f deploy/docker-compose.yml down
