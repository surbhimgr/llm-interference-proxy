// LLM Inference Proxy — main entry point
//
// Environment variables:
//   GRPC_PORT           — gRPC server port (default: 50051)
//   METRICS_PORT        — Prometheus metrics HTTP port (default: 9090)
//   REDIS_ADDR          — Redis address (default: localhost:6379)
//   REDIS_PASSWORD      — Redis password (default: "")
//   REDIS_DB            — Redis database (default: 0)
//   CACHE_TTL           — Cache TTL duration (default: 1h)
//   QDRANT_URL          — Qdrant server URL (default: http://localhost:6333)
//   QDRANT_COLLECTION   — Qdrant collection name (default: llm_cache)
//   SIMILARITY_THRESHOLD — Semantic similarity threshold (default: 0.95)
//   EMBEDDING_API_KEY   — API key for the embedding provider
//   OPENAI_API_KEYS     — Comma-separated OpenAI API keys
//   GEMINI_API_KEYS     — Comma-separated Gemini API keys
//   REQUEST_TIMEOUT     — Request timeout duration (default: 30s)
//   MAX_RETRIES         — Maximum retry attempts (default: 3)
//   CB_FAILURE_THRESHOLD — Circuit breaker failure threshold (default: 5)
//   CB_COOLDOWN         — Circuit breaker cooldown (default: 30s)
package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	pb "github.com/abdhe/llm-inference-proxy/proto"
	"github.com/abdhe/llm-inference-proxy/pkg/cache"
	"github.com/abdhe/llm-inference-proxy/pkg/provider"
	"github.com/abdhe/llm-inference-proxy/pkg/proxy"
	"github.com/abdhe/llm-inference-proxy/pkg/resilience"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.Println("Starting LLM Inference Proxy...")

	// -------------------------------------------------------------------------
	// Configuration from environment
	// -------------------------------------------------------------------------
	grpcPort := envOrDefault("GRPC_PORT", "50051")
	metricsPort := envOrDefault("METRICS_PORT", "9090")
	redisAddr := envOrDefault("REDIS_ADDR", "localhost:6379")
	redisPassword := envOrDefault("REDIS_PASSWORD", "")
	redisDB := envIntOrDefault("REDIS_DB", 0)
	cacheTTL := envDurationOrDefault("CACHE_TTL", 1*time.Hour)
	qdrantURL := envOrDefault("QDRANT_URL", "http://localhost:6333")
	qdrantCollection := envOrDefault("QDRANT_COLLECTION", "llm_cache")
	similarityThreshold := envFloatOrDefault("SIMILARITY_THRESHOLD", 0.95)
	embeddingAPIKey := os.Getenv("EMBEDDING_API_KEY")
	openaiKeys := splitKeys(os.Getenv("OPENAI_API_KEYS"))
	geminiKeys := splitKeys(os.Getenv("GEMINI_API_KEYS"))
	requestTimeout := envDurationOrDefault("REQUEST_TIMEOUT", 30*time.Second)
	maxRetries := envIntOrDefault("MAX_RETRIES", 3)
	cbFailureThreshold := envIntOrDefault("CB_FAILURE_THRESHOLD", 5)
	cbCooldown := envDurationOrDefault("CB_COOLDOWN", 30*time.Second)

	// -------------------------------------------------------------------------
	// Initialize providers
	// -------------------------------------------------------------------------
	providers := map[string]provider.Provider{
		"openai": provider.NewOpenAIProvider(),
		"gemini": provider.NewGeminiProvider(),
	}

	// -------------------------------------------------------------------------
	// Initialize key pools
	// -------------------------------------------------------------------------
	keyPools := make(map[string]*resilience.KeyPool)
	if len(openaiKeys) > 0 {
		keyPools["openai"] = resilience.NewKeyPool(openaiKeys)
		log.Printf("OpenAI key pool: %d keys", len(openaiKeys))
	}
	if len(geminiKeys) > 0 {
		keyPools["gemini"] = resilience.NewKeyPool(geminiKeys)
		log.Printf("Gemini key pool: %d keys", len(geminiKeys))
	}

	// -------------------------------------------------------------------------
	// Initialize circuit breakers
	// -------------------------------------------------------------------------
	cbCfg := resilience.CircuitBreakerConfig{
		FailureThreshold: cbFailureThreshold,
		Cooldown:         cbCooldown,
	}
	circuitBreakers := map[string]*resilience.CircuitBreaker{
		"openai": resilience.NewCircuitBreaker(cbCfg),
		"gemini": resilience.NewCircuitBreaker(cbCfg),
	}

	// -------------------------------------------------------------------------
	// Initialize semantic cache
	// -------------------------------------------------------------------------
	var semanticCache *cache.SemanticCache
	if embeddingAPIKey != "" {
		embedder := cache.NewEmbedder(embeddingAPIKey)
		vectorStore := cache.NewVectorStore(qdrantURL, qdrantCollection)
		redisCache := cache.NewRedisCache(redisAddr, redisPassword, redisDB, cacheTTL)

		// Verify Redis connection
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := redisCache.Ping(ctx); err != nil {
			log.Printf("WARNING: Redis connection failed: %v (cache disabled)", err)
		} else {
			semanticCache = cache.NewSemanticCache(embedder, vectorStore, redisCache, float32(similarityThreshold))
			log.Printf("Semantic cache enabled (threshold=%.2f, TTL=%s)", similarityThreshold, cacheTTL)
		}
		cancel()
	} else {
		log.Println("WARNING: EMBEDDING_API_KEY not set — semantic cache disabled")
	}

	// -------------------------------------------------------------------------
	// Initialize retry config
	// -------------------------------------------------------------------------
	retryCfg := resilience.RetryConfig{
		MaxRetries: maxRetries,
		BaseDelay:  500 * time.Millisecond,
		MaxDelay:   30 * time.Second,
	}

	// -------------------------------------------------------------------------
	// Create gRPC handler
	// -------------------------------------------------------------------------
	handler := proxy.NewHandler(proxy.Config{
		Providers:       providers,
		KeyPools:        keyPools,
		CircuitBreakers: circuitBreakers,
		SemanticCache:   semanticCache,
		RetryConfig:     retryCfg,
		RequestTimeout:  requestTimeout,
	})

	// -------------------------------------------------------------------------
	// Start gRPC server
	// -------------------------------------------------------------------------
	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(4*1024*1024),    // 4MB
		grpc.MaxSendMsgSize(16*1024*1024),   // 16MB
	)
	pb.RegisterInferenceServiceServer(grpcServer, handler)
	reflection.Register(grpcServer) // Enable gRPC reflection for grpcurl

	grpcLis, err := net.Listen("tcp", ":"+grpcPort)
	if err != nil {
		log.Fatalf("Failed to listen on gRPC port %s: %v", grpcPort, err)
	}

	go func() {
		log.Printf("gRPC server listening on :%s", grpcPort)
		if err := grpcServer.Serve(grpcLis); err != nil {
			log.Fatalf("gRPC server error: %v", err)
		}
	}()

	// -------------------------------------------------------------------------
	// Start HTTP metrics server
	// -------------------------------------------------------------------------
	metricsMux := http.NewServeMux()
	metricsMux.Handle("/metrics", promhttp.Handler())
	metricsMux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "ok")
	})

	metricsServer := &http.Server{
		Addr:         ":" + metricsPort,
		Handler:      metricsMux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	go func() {
		log.Printf("Metrics server listening on :%s/metrics", metricsPort)
		if err := metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Metrics server error: %v", err)
		}
	}()

	// -------------------------------------------------------------------------
	// Graceful shutdown
	// -------------------------------------------------------------------------
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	sig := <-sigCh
	log.Printf("Received signal %v, shutting down...", sig)

	// Gracefully stop gRPC server
	grpcServer.GracefulStop()
	log.Println("gRPC server stopped")

	// Shut down metrics server
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()
	if err := metricsServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("Metrics server shutdown error: %v", err)
	}
	log.Println("Metrics server stopped")

	log.Println("LLM Inference Proxy shut down successfully")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func envOrDefault(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

func envIntOrDefault(key string, defaultVal int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return defaultVal
}

func envFloatOrDefault(key string, defaultVal float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return defaultVal
}

func envDurationOrDefault(key string, defaultVal time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return defaultVal
}

func splitKeys(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	var keys []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			keys = append(keys, p)
		}
	}
	return keys
}
