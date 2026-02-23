# ============================================================================
# Stage 1: Build
# ============================================================================
FROM golang:1.22-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /build

# Cache dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy source and build
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-s -w" \
    -o /build/llm-proxy \
    ./cmd/proxy

# ============================================================================
# Stage 2: Runtime (distroless for minimal attack surface)
# ============================================================================
FROM gcr.io/distroless/static-debian12

COPY --from=builder /build/llm-proxy /llm-proxy
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# gRPC port
EXPOSE 50051
# Prometheus metrics port
EXPOSE 9090

USER nonroot:nonroot

ENTRYPOINT ["/llm-proxy"]
