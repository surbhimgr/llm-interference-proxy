package resilience

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"
)

// RetryConfig holds configuration for the exponential backoff retry logic.
type RetryConfig struct {
	MaxRetries int           // Maximum number of retry attempts
	BaseDelay  time.Duration // Initial delay before first retry
	MaxDelay   time.Duration // Maximum delay cap
}

// DefaultRetryConfig returns sensible defaults for retry configuration.
func DefaultRetryConfig() RetryConfig {
	return RetryConfig{
		MaxRetries: 3,
		BaseDelay:  500 * time.Millisecond,
		MaxDelay:   30 * time.Second,
	}
}

// RetryableFunc is a function that can be retried.
// It should return a non-nil error to trigger a retry.
type RetryableFunc func(ctx context.Context) error

// Retry executes fn with exponential backoff and full jitter.
// delay = rand(0, min(maxDelay, baseDelay * 2^attempt))
// It respects context cancellation at every step.
func Retry(ctx context.Context, cfg RetryConfig, fn RetryableFunc) error {
	var lastErr error

	for attempt := 0; attempt <= cfg.MaxRetries; attempt++ {
		// Check context before each attempt
		select {
		case <-ctx.Done():
			return fmt.Errorf("retry: context cancelled: %w", ctx.Err())
		default:
		}

		lastErr = fn(ctx)
		if lastErr == nil {
			return nil
		}

		// Don't sleep after the last attempt
		if attempt == cfg.MaxRetries {
			break
		}

		// Only retry on server errors (5xx, 429)
		if !IsServerError(lastErr) {
			return lastErr
		}

		// Calculate delay with full jitter
		delay := calculateDelay(attempt, cfg.BaseDelay, cfg.MaxDelay)

		select {
		case <-ctx.Done():
			return fmt.Errorf("retry: context cancelled during backoff: %w", ctx.Err())
		case <-time.After(delay):
			// Continue to next attempt
		}
	}

	return fmt.Errorf("retry: max retries (%d) exceeded: %w", cfg.MaxRetries, lastErr)
}

// calculateDelay computes the jittered backoff delay.
// Uses "Full Jitter": delay = rand(0, min(cap, base * 2^attempt))
func calculateDelay(attempt int, baseDelay, maxDelay time.Duration) time.Duration {
	// base * 2^attempt
	expDelay := float64(baseDelay) * math.Pow(2, float64(attempt))

	// Cap at maxDelay
	if expDelay > float64(maxDelay) {
		expDelay = float64(maxDelay)
	}

	// Full jitter: uniform random in [0, expDelay)
	jitteredDelay := time.Duration(rand.Float64() * expDelay)

	// Ensure at least 1ms
	if jitteredDelay < time.Millisecond {
		jitteredDelay = time.Millisecond
	}

	return jitteredDelay
}
