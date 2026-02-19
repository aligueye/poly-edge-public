FROM golang:1.24-alpine AS builder
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /poly-edge ./cmd/poly-edge

FROM alpine:3.20
RUN apk add --no-cache ca-certificates wget && \
    adduser -D -u 1000 polyedge
COPY --from=builder /poly-edge /poly-edge
USER polyedge
ENTRYPOINT ["/poly-edge"]
