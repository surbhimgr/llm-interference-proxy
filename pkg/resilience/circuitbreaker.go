package resilience

import (
	"errors"
	"sync"
	"time"
)

// CircuitState represents the state of a circuit breaker.
type CircuitState int

const (
	StateClosed   CircuitState = iota // Normal — requests pass through
	StateOpen                         // Tripped — requests are rejected
	StateHalfOpen                     // Probing — one request allowed
)

// ErrCircuitOpen is returned when the circuit breaker is open.
var ErrCircuitOpen = errors.New("circuit breaker is open")

// CircuitBreaker implements the circuit breaker pattern.
// It trips open after consecutive failures exceed a threshold, and
// transitions to half-open after a cooldown period.
type CircuitBreaker struct {
	mu sync.Mutex

	state              CircuitState
	failureThreshold   int
	consecutiveFailures int
	cooldown           time.Duration
	lastFailure        time.Time

	// Counters for observability
	totalSuccesses int64
	totalFailures  int64
	totalRejected  int64
}

// CircuitBreakerConfig holds configuration for a CircuitBreaker.
type CircuitBreakerConfig struct {
	FailureThreshold int           // Number of consecutive failures to trip
	Cooldown         time.Duration // Time to wait before probing
}

// NewCircuitBreaker creates a new circuit breaker with the given config.
func NewCircuitBreaker(cfg CircuitBreakerConfig) *CircuitBreaker {
	if cfg.FailureThreshold <= 0 {
		cfg.FailureThreshold = 5
	}
	if cfg.Cooldown <= 0 {
		cfg.Cooldown = 30 * time.Second
	}

	return &CircuitBreaker{
		state:            StateClosed,
		failureThreshold: cfg.FailureThreshold,
		cooldown:         cfg.Cooldown,
	}
}

// Execute runs the given function through the circuit breaker.
// Returns ErrCircuitOpen if the circuit is open and cooldown hasn't elapsed.
func (cb *CircuitBreaker) Execute(fn func() error) error {
	if !cb.allowRequest() {
		cb.mu.Lock()
		cb.totalRejected++
		cb.mu.Unlock()
		return ErrCircuitOpen
	}

	err := fn()

	cb.mu.Lock()
	defer cb.mu.Unlock()

	if err != nil {
		cb.recordFailure()
		return err
	}

	cb.recordSuccess()
	return nil
}

// State returns the current state of the circuit breaker.
func (cb *CircuitBreaker) State() CircuitState {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	// Check for automatic transition from open to half-open
	if cb.state == StateOpen && time.Since(cb.lastFailure) > cb.cooldown {
		return StateHalfOpen
	}
	return cb.state
}

// allowRequest checks whether a request is allowed.
func (cb *CircuitBreaker) allowRequest() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case StateClosed:
		return true
	case StateOpen:
		// Check if cooldown has elapsed → transition to half-open
		if time.Since(cb.lastFailure) > cb.cooldown {
			cb.state = StateHalfOpen
			return true
		}
		return false
	case StateHalfOpen:
		// Allow one probe request
		return true
	default:
		return false
	}
}

// recordFailure records a failed call. Must be called with mu held.
func (cb *CircuitBreaker) recordFailure() {
	cb.consecutiveFailures++
	cb.totalFailures++
	cb.lastFailure = time.Now()

	if cb.consecutiveFailures >= cb.failureThreshold {
		cb.state = StateOpen
	}
}

// recordSuccess records a successful call. Must be called with mu held.
func (cb *CircuitBreaker) recordSuccess() {
	cb.totalSuccesses++
	cb.consecutiveFailures = 0

	// Any success resets the circuit to closed
	if cb.state == StateHalfOpen {
		cb.state = StateClosed
	}
}

// IsServerError returns true if the error represents a 5xx or 429 error
// that should count as a circuit breaker failure.
func IsServerError(err error) bool {
	if err == nil {
		return false
	}
	// Check if the error message contains status codes we care about
	errMsg := err.Error()
	for _, code := range []string{"429", "500", "502", "503", "504"} {
		if contains(errMsg, code) {
			return true
		}
	}
	return false
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
