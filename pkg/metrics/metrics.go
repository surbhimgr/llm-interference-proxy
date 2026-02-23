// Package metrics provides Prometheus instrumentation for the proxy.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// RequestLatency tracks end-to-end request latency in seconds.
	RequestLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "request_latency_seconds",
			Help:    "End-to-end request latency in seconds.",
			Buckets: []float64{0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30},
		},
		[]string{"provider", "model", "cache_status"},
	)

	// TokenUsageTotal tracks the total number of tokens consumed.
	TokenUsageTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "token_usage_total",
			Help: "Total number of tokens consumed.",
		},
		[]string{"provider", "model", "direction"}, // direction: "input" or "output"
	)

	// CacheHitsTotal tracks the total number of cache hits.
	CacheHitsTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "cache_hits_total",
			Help: "Total number of semantic cache hits.",
		},
	)

	// CacheLookupsTotal tracks the total number of cache lookups.
	CacheLookupsTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "cache_lookups_total",
			Help: "Total number of semantic cache lookups.",
		},
	)

	// CacheHitRatio is a summary gauge that's computed from hits/lookups.
	// This is exposed as a derived metric: cache_hits_total / cache_lookups_total.
	// Prometheus can compute this in queries, but we also expose it as a gauge
	// for convenience.
	CacheHitRatio = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "cache_hit_ratio",
			Help: "Current cache hit ratio (hits / lookups). Computed per-update.",
		},
	)

	// CircuitBreakerState tracks the current state of each circuit breaker.
	CircuitBreakerState = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "circuit_breaker_state",
			Help: "Current circuit breaker state: 0=closed, 1=open, 2=half-open.",
		},
		[]string{"provider"},
	)

	// ActiveRequests tracks the number of currently in-flight requests.
	ActiveRequests = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "active_requests",
			Help: "Number of currently in-flight requests.",
		},
	)

	// RequestsTotal tracks total requests by status.
	RequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "requests_total",
			Help: "Total number of requests by status.",
		},
		[]string{"status"}, // "success", "error", "cache_hit"
	)

	// trackingMu guards the ratio update — not needed since gauge.Set is atomic
	totalHits    float64
	totalLookups float64
)

// RecordCacheLookup records a cache lookup and updates the hit ratio.
func RecordCacheLookup(hit bool) {
	CacheLookupsTotal.Inc()
	totalLookups++

	if hit {
		CacheHitsTotal.Inc()
		totalHits++
	}

	// Update ratio gauge
	if totalLookups > 0 {
		CacheHitRatio.Set(totalHits / totalLookups)
	}
}
