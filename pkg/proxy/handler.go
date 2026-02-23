// Package proxy implements the gRPC server handler for inference requests.
package proxy

import (
	"context"
	"fmt"
	"log"
	"time"

	pb "github.com/abdhe/llm-inference-proxy/proto"
	"github.com/abdhe/llm-inference-proxy/pkg/cache"
	"github.com/abdhe/llm-inference-proxy/pkg/metrics"
	"github.com/abdhe/llm-inference-proxy/pkg/provider"
	"github.com/abdhe/llm-inference-proxy/pkg/resilience"
)

// Handler implements the gRPC InferenceServiceServer.
type Handler struct {
	pb.UnimplementedInferenceServiceServer

	providers      map[string]provider.Provider // model-prefix → provider
	keyPools       map[string]*resilience.KeyPool
	circuitBreakers map[string]*resilience.CircuitBreaker
	semanticCache  *cache.SemanticCache
	retryCfg       resilience.RetryConfig
	requestTimeout time.Duration
}

// Config holds the handler configuration.
type Config struct {
	Providers       map[string]provider.Provider
	KeyPools        map[string]*resilience.KeyPool
	CircuitBreakers map[string]*resilience.CircuitBreaker
	SemanticCache   *cache.SemanticCache
	RetryConfig     resilience.RetryConfig
	RequestTimeout  time.Duration
}

// NewHandler creates a new proxy handler.
func NewHandler(cfg Config) *Handler {
	if cfg.RequestTimeout == 0 {
		cfg.RequestTimeout = 30 * time.Second
	}
	return &Handler{
		providers:       cfg.Providers,
		keyPools:        cfg.KeyPools,
		circuitBreakers: cfg.CircuitBreakers,
		semanticCache:   cfg.SemanticCache,
		retryCfg:        cfg.RetryConfig,
		requestTimeout:  cfg.RequestTimeout,
	}
}

// Infer handles a unary inference request.
func (h *Handler) Infer(ctx context.Context, req *pb.InferenceRequest) (*pb.InferenceResponse, error) {
	start := time.Now()
	metrics.ActiveRequests.Inc()
	defer metrics.ActiveRequests.Dec()

	// Apply timeout
	ctx, cancel := context.WithTimeout(ctx, h.requestTimeout)
	defer cancel()

	providerName := resolveProvider(req.Model)

	// -------------------------------------------------------------------------
	// Step 1: Semantic cache lookup
	// -------------------------------------------------------------------------
	if h.semanticCache != nil {
		metrics.CacheLookupsTotal.Inc()
		cacheResult, err := h.semanticCache.Lookup(ctx, req.Prompt)
		if err != nil {
			log.Printf("[proxy] cache lookup error: %v", err)
		}

		if cacheResult.Hit {
			metrics.RecordCacheLookup(true)
			metrics.RequestsTotal.WithLabelValues("cache_hit").Inc()

			latency := time.Since(start)
			metrics.RequestLatency.WithLabelValues(providerName, req.Model, "hit").Observe(latency.Seconds())

			return &pb.InferenceResponse{
				Text:         cacheResult.Response.Text,
				PromptTokens: cacheResult.Response.PromptTokens,
				OutputTokens: cacheResult.Response.OutputTokens,
				CacheHit:     true,
				LatencyMs:    float64(latency.Milliseconds()),
			}, nil
		}
		metrics.RecordCacheLookup(false)
	}

	// -------------------------------------------------------------------------
	// Step 2: Resolve provider
	// -------------------------------------------------------------------------
	p, ok := h.providers[providerName]
	if !ok {
		return nil, fmt.Errorf("unknown provider for model %q", req.Model)
	}

	// -------------------------------------------------------------------------
	// Step 3: Get API key from pool
	// -------------------------------------------------------------------------
	kp, ok := h.keyPools[providerName]
	if !ok {
		return nil, fmt.Errorf("no key pool for provider %q", providerName)
	}

	apiKey, err := kp.Next()
	if err != nil {
		return nil, fmt.Errorf("key pool: %w", err)
	}

	// -------------------------------------------------------------------------
	// Step 4: Execute with circuit breaker + retry
	// -------------------------------------------------------------------------
	provReq := provider.Request{
		Model:       req.Model,
		Prompt:      req.Prompt,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		APIKey:      apiKey,
	}

	var resp provider.Response

	cb := h.circuitBreakers[providerName]
	if cb == nil {
		// No circuit breaker — execute directly with retry
		err = resilience.Retry(ctx, h.retryCfg, func(ctx context.Context) error {
			var retryErr error
			resp, retryErr = p.Infer(ctx, provReq)
			return retryErr
		})
	} else {
		// Circuit breaker wrapping retry
		err = cb.Execute(func() error {
			return resilience.Retry(ctx, h.retryCfg, func(ctx context.Context) error {
				var retryErr error
				resp, retryErr = p.Infer(ctx, provReq)
				return retryErr
			})
		})

		// Update circuit breaker metric
		metrics.CircuitBreakerState.WithLabelValues(providerName).Set(float64(cb.State()))
	}

	if err != nil {
		// Mark key rate-limited if it's a 429
		if resilience.IsServerError(err) {
			kp.MarkRateLimited(apiKey, time.Now().Add(60*time.Second))
		}

		metrics.RequestsTotal.WithLabelValues("error").Inc()
		latency := time.Since(start)
		metrics.RequestLatency.WithLabelValues(providerName, req.Model, "error").Observe(latency.Seconds())

		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// -------------------------------------------------------------------------
	// Step 5: Record metrics
	// -------------------------------------------------------------------------
	latency := time.Since(start)
	metrics.RequestLatency.WithLabelValues(providerName, req.Model, "miss").Observe(latency.Seconds())
	metrics.TokenUsageTotal.WithLabelValues(providerName, req.Model, "input").Add(float64(resp.PromptTokens))
	metrics.TokenUsageTotal.WithLabelValues(providerName, req.Model, "output").Add(float64(resp.OutputTokens))
	metrics.RequestsTotal.WithLabelValues("success").Inc()

	// -------------------------------------------------------------------------
	// Step 6: Store in semantic cache (async, non-blocking)
	// -------------------------------------------------------------------------
	if h.semanticCache != nil {
		go h.semanticCache.Store(context.Background(), req.Prompt, resp)
	}

	return &pb.InferenceResponse{
		Text:         resp.Text,
		PromptTokens: resp.PromptTokens,
		OutputTokens: resp.OutputTokens,
		CacheHit:     false,
		LatencyMs:    float64(latency.Milliseconds()),
	}, nil
}

// InferStream handles a server-side streaming inference request.
func (h *Handler) InferStream(req *pb.InferenceRequest, stream pb.InferenceService_InferStreamServer) error {
	start := time.Now()
	metrics.ActiveRequests.Inc()
	defer metrics.ActiveRequests.Dec()

	ctx := stream.Context()
	ctx, cancel := context.WithTimeout(ctx, h.requestTimeout)
	defer cancel()

	providerName := resolveProvider(req.Model)

	// -------------------------------------------------------------------------
	// Step 1: Check cache (streaming requests can still return cached results)
	// -------------------------------------------------------------------------
	if h.semanticCache != nil {
		metrics.CacheLookupsTotal.Inc()
		cacheResult, _ := h.semanticCache.Lookup(ctx, req.Prompt)
		if cacheResult.Hit {
			metrics.RecordCacheLookup(true)
			metrics.RequestsTotal.WithLabelValues("cache_hit").Inc()

			latency := time.Since(start)
			metrics.RequestLatency.WithLabelValues(providerName, req.Model, "hit").Observe(latency.Seconds())

			// Send the full cached response as a single chunk
			return stream.Send(&pb.StreamChunk{
				Text:         cacheResult.Response.Text,
				Done:         true,
				PromptTokens: cacheResult.Response.PromptTokens,
				OutputTokens: cacheResult.Response.OutputTokens,
			})
		}
		metrics.RecordCacheLookup(false)
	}

	// -------------------------------------------------------------------------
	// Step 2: Resolve provider + key
	// -------------------------------------------------------------------------
	p, ok := h.providers[providerName]
	if !ok {
		return fmt.Errorf("unknown provider for model %q", req.Model)
	}

	kp, ok := h.keyPools[providerName]
	if !ok {
		return fmt.Errorf("no key pool for provider %q", providerName)
	}

	apiKey, err := kp.Next()
	if err != nil {
		return fmt.Errorf("key pool: %w", err)
	}

	provReq := provider.Request{
		Model:       req.Model,
		Prompt:      req.Prompt,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		APIKey:      apiKey,
	}

	// -------------------------------------------------------------------------
	// Step 3: Stream from provider
	// -------------------------------------------------------------------------
	chunks, err := p.InferStream(ctx, provReq)
	if err != nil {
		metrics.RequestsTotal.WithLabelValues("error").Inc()
		return fmt.Errorf("stream inference failed: %w", err)
	}

	var fullText string
	var promptTokens, outputTokens int32

	for chunk := range chunks {
		if chunk.Err != nil {
			return fmt.Errorf("stream chunk error: %w", chunk.Err)
		}

		fullText += chunk.Text
		if chunk.PromptTokens > 0 {
			promptTokens = chunk.PromptTokens
		}
		if chunk.OutputTokens > 0 {
			outputTokens = chunk.OutputTokens
		}

		if err := stream.Send(&pb.StreamChunk{
			Text:         chunk.Text,
			Done:         chunk.Done,
			PromptTokens: chunk.PromptTokens,
			OutputTokens: chunk.OutputTokens,
		}); err != nil {
			return fmt.Errorf("stream send: %w", err)
		}
	}

	// -------------------------------------------------------------------------
	// Step 4: Record metrics + cache
	// -------------------------------------------------------------------------
	latency := time.Since(start)
	metrics.RequestLatency.WithLabelValues(providerName, req.Model, "miss").Observe(latency.Seconds())
	metrics.TokenUsageTotal.WithLabelValues(providerName, req.Model, "input").Add(float64(promptTokens))
	metrics.TokenUsageTotal.WithLabelValues(providerName, req.Model, "output").Add(float64(outputTokens))
	metrics.RequestsTotal.WithLabelValues("success").Inc()

	// Cache the full assembled response
	if h.semanticCache != nil && fullText != "" {
		go h.semanticCache.Store(context.Background(), req.Prompt, provider.Response{
			Text:         fullText,
			PromptTokens: promptTokens,
			OutputTokens: outputTokens,
		})
	}

	return nil
}

// resolveProvider maps a model name to a provider name.
func resolveProvider(model string) string {
	// Simple prefix-based routing
	switch {
	case len(model) >= 3 && model[:3] == "gpt":
		return "openai"
	case len(model) >= 6 && model[:6] == "gemini":
		return "gemini"
	case len(model) >= 7 && model[:7] == "claude-":
		return "anthropic"
	default:
		return "openai" // default fallback
	}
}
