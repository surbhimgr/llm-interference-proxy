<![CDATA[# ⚡ LLM Inference Proxy

A high-throughput, low-latency inference proxy for Large Language Models, built in **Go**. It sits between your application and LLM providers (OpenAI, Gemini), adding **semantic caching**, **resilience patterns**, and **observability** — all exposed over a single **gRPC** API.

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │           LLM Inference Proxy               │
                         │                                             │
  gRPC client ──────────►│  Handler ──► Semantic Cache                 │
  (Infer / InferStream)  │      │        ├─ Embedder  (embedding API)  │
                         │      │        ├─ Qdrant    (vector search)  │
                         │      │        └─ Redis     (response store) │
                         │      │                                      │
                         │      ├──► Key Pool (round-robin rotation)   │
                         │      ├──► Circuit Breaker (per provider)    │
                         │      ├──► Retry w/ Exponential Backoff      │
                         │      └──► Provider                          │
                         │            ├─ OpenAI                        │
                         │            └─ Gemini                        │
                         │                                             │
                         │  :9090/metrics  ◄── Prometheus              │
                         └─────────────────────────────────────────────┘
```

---

## Features

| Category | Details |
|---|---|
| **Transport** | gRPC with **unary** (`Infer`) and **server-streaming** (`InferStream`) RPCs |
| **Providers** | OpenAI and Google Gemini, behind a pluggable `Provider` interface |
| **Semantic Cache** | Embed prompts → vector-search in **Qdrant** → store/retrieve responses in **Redis**. Configurable similarity threshold |
| **Key Pool** | Round-robin API key rotation with per-key rate-limit tracking and automatic reset |
| **Circuit Breaker** | Per-provider; trips after *N* consecutive failures, transitions through Closed → Open → Half-Open |
| **Retry** | Exponential backoff with **full jitter**, retries only on 5xx / 429 errors |
| **Observability** | Prometheus metrics — latency histograms, token counters, cache-hit ratio, circuit-breaker state, active requests |
| **Infrastructure** | Multi-stage Dockerfile (distroless runtime), Kubernetes Deployment + Service + HPA |

---

## Project Structure

```
llm_inference_proxy/
├── cmd/proxy/main.go          # Entry point — wires everything together
├── proto/
│   ├── proxy.proto            # gRPC service & message definitions
│   ├── proxy.pb.go            # Generated protobuf code
│   └── proxy_grpc.pb.go       # Generated gRPC stubs
├── pkg/
│   ├── provider/
│   │   ├── provider.go        # Provider interface + shared types
│   │   ├── openai.go          # OpenAI HTTP provider
│   │   └── gemini.go          # Google Gemini HTTP provider
│   ├── cache/
│   │   ├── semantic_cache.go  # Embed → search → hit/miss orchestrator
│   │   ├── embedder.go        # Embedding API client
│   │   ├── vector_store.go    # Qdrant vector DB client
│   │   └── redis_cache.go     # Redis response cache
│   ├── resilience/
│   │   ├── keypool.go         # Virtual key pool with rate-limit awareness
│   │   ├── circuitbreaker.go  # Circuit breaker (Closed/Open/Half-Open)
│   │   └── retry.go           # Exponential backoff + full jitter
│   ├── proxy/
│   │   └── handler.go         # gRPC handler (Infer + InferStream)
│   └── metrics/
│       └── metrics.go         # Prometheus counters, histograms, gauges
├── k8s/deployment.yaml        # Deployment + Service + HPA (GKE-optimized)
├── Dockerfile                 # Multi-stage build (Alpine → Distroless)
├── go.mod
└── go.sum
```

---

## Getting Started

### Prerequisites

- **Go 1.22+**
- **Redis** — response cache storage
- **Qdrant** — vector database for semantic search
- **protoc** + Go plugins *(only if regenerating proto)*

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GRPC_PORT` | `50051` | gRPC server port |
| `METRICS_PORT` | `9090` | Prometheus metrics HTTP port |
| `REDIS_ADDR` | `localhost:6379` | Redis address |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST endpoint |
| `QDRANT_COLLECTION` | `llm_cache` | Qdrant collection name |
| `SIMILARITY_THRESHOLD` | `0.95` | Cosine-similarity threshold for cache hits |
| `CACHE_TTL` | `1h` | Redis cache TTL |
| `REQUEST_TIMEOUT` | `30s` | Per-request context timeout |
| `MAX_RETRIES` | `3` | Max retry attempts |
| `CB_FAILURE_THRESHOLD` | `5` | Consecutive failures to trip circuit |
| `CB_COOLDOWN` | `30s` | Cooldown before half-open probe |
| `EMBEDDING_API_KEY` | — | API key for embedding model |
| `OPENAI_API_KEYS` | — | Comma-separated OpenAI API keys |
| `GEMINI_API_KEYS` | — | Comma-separated Gemini API keys |

### Run Locally

```bash
# Start dependencies
# (Redis at :6379, Qdrant at :6333)

# Set API keys
export OPENAI_API_KEYS="sk-key1,sk-key2"
export GEMINI_API_KEYS="AIza..."
export EMBEDDING_API_KEY="your-embedding-key"

# Build & run
go build -o llm-proxy ./cmd/proxy
./llm-proxy
```

The proxy listens on **:50051** (gRPC) and **:9090** (metrics + health).

### Test with grpcurl

```bash
# Unary inference
grpcurl -plaintext -d '{
  "model": "gpt-4",
  "prompt": "Explain goroutines in one paragraph.",
  "temperature": 0.7,
  "max_tokens": 256
}' localhost:50051 inferenceproxy.InferenceService/Infer

# Streaming inference
grpcurl -plaintext -d '{
  "model": "gemini-pro",
  "prompt": "Write a haiku about concurrency.",
  "temperature": 0.9,
  "max_tokens": 64
}' localhost:50051 inferenceproxy.InferenceService/InferStream
```

---

## Docker

```bash
# Build
docker build -t llm-inference-proxy .

# Run
docker run -p 50051:50051 -p 9090:9090 \
  -e OPENAI_API_KEYS="sk-..." \
  -e GEMINI_API_KEYS="AIza..." \
  -e EMBEDDING_API_KEY="..." \
  -e REDIS_ADDR="host.docker.internal:6379" \
  -e QDRANT_URL="http://host.docker.internal:6333" \
  llm-inference-proxy
```

The image uses a **distroless** base for a minimal attack surface (~10 MB final image).

---

## Kubernetes

The `k8s/deployment.yaml` provides a production-ready setup:

- **Deployment** — 2 replicas, resource limits (128 Mi / 500 m), health probes (startup, readiness, liveness)
- **Service** — ClusterIP exposing gRPC (:50051) and metrics (:9090)
- **HPA** — Scales 2 → 20 pods based on CPU (70%) and memory (80%) utilization

```bash
# Create secrets
kubectl create secret generic llm-proxy-secrets \
  --from-literal=openai-api-keys="sk-..." \
  --from-literal=gemini-api-keys="AIza..." \
  --from-literal=embedding-api-key="..."

# Deploy
kubectl apply -f k8s/deployment.yaml
```

---

## Observability

Prometheus metrics are served at `GET :9090/metrics`:

| Metric | Type | Labels | Description |
|---|---|---|---|
| `request_latency_seconds` | Histogram | `provider`, `model`, `cache_status` | End-to-end latency |
| `token_usage_total` | Counter | `provider`, `model`, `direction` | Tokens consumed (input / output) |
| `cache_hits_total` | Counter | — | Semantic cache hits |
| `cache_lookups_total` | Counter | — | Total cache lookups |
| `cache_hit_ratio` | Gauge | — | Live hit ratio |
| `circuit_breaker_state` | Gauge | `provider` | 0 = closed, 1 = open, 2 = half-open |
| `active_requests` | Gauge | — | In-flight requests |
| `requests_total` | Counter | `status` | Requests by outcome |

A `/healthz` endpoint is also available on the metrics port for liveness/readiness probes.

---

## Proto Definition

```protobuf
service InferenceService {
  rpc Infer(InferenceRequest) returns (InferenceResponse);
  rpc InferStream(InferenceRequest) returns (stream StreamChunk);
}
```

Regenerate Go stubs:

```bash
protoc --go_out=. --go-grpc_out=. proto/proxy.proto
```

---

## License

This project is provided as-is for educational and portfolio purposes.
]]>
