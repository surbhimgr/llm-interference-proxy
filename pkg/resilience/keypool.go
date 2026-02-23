// Package resilience provides resiliency patterns for the proxy.
package resilience

import (
	"fmt"
	"sync"
	"time"
)

// KeyPool manages a pool of API keys with round-robin rotation
// and per-key rate-limit awareness.
type KeyPool struct {
	mu      sync.Mutex
	keys    []keyEntry
	current int
}

type keyEntry struct {
	Key       string
	Remaining int       // Remaining calls before rate limit
	ResetAt   time.Time // When the rate limit resets
	Exhausted bool      // Temporarily exhausted
}

// NewKeyPool creates a key pool from a list of API keys.
func NewKeyPool(keys []string) *KeyPool {
	entries := make([]keyEntry, len(keys))
	for i, k := range keys {
		entries[i] = keyEntry{
			Key:       k,
			Remaining: -1, // Unknown initially
		}
	}
	return &KeyPool{keys: entries}
}

// Next returns the next available API key using round-robin selection.
// It skips keys that are currently exhausted (rate-limited).
// Returns an error if all keys are exhausted.
func (kp *KeyPool) Next() (string, error) {
	kp.mu.Lock()
	defer kp.mu.Unlock()

	n := len(kp.keys)
	if n == 0 {
		return "", fmt.Errorf("keypool: no keys configured")
	}

	now := time.Now()

	// Try each key once in round-robin order
	for i := 0; i < n; i++ {
		idx := (kp.current + i) % n
		entry := &kp.keys[idx]

		// Reset exhausted keys whose cooldown has passed
		if entry.Exhausted && now.After(entry.ResetAt) {
			entry.Exhausted = false
			entry.Remaining = -1
		}

		if !entry.Exhausted {
			kp.current = (idx + 1) % n
			return entry.Key, nil
		}
	}

	// All keys exhausted — find the earliest reset time
	earliest := kp.keys[0].ResetAt
	for _, e := range kp.keys[1:] {
		if e.ResetAt.Before(earliest) {
			earliest = e.ResetAt
		}
	}

	return "", fmt.Errorf("keypool: all keys exhausted, earliest reset at %s", earliest.Format(time.RFC3339))
}

// MarkRateLimited marks a key as rate-limited with the given reset time.
func (kp *KeyPool) MarkRateLimited(key string, resetAt time.Time) {
	kp.mu.Lock()
	defer kp.mu.Unlock()

	for i := range kp.keys {
		if kp.keys[i].Key == key {
			kp.keys[i].Exhausted = true
			kp.keys[i].ResetAt = resetAt
			kp.keys[i].Remaining = 0
			return
		}
	}
}

// UpdateRemaining updates the remaining call count for a key.
func (kp *KeyPool) UpdateRemaining(key string, remaining int, resetAt time.Time) {
	kp.mu.Lock()
	defer kp.mu.Unlock()

	for i := range kp.keys {
		if kp.keys[i].Key == key {
			kp.keys[i].Remaining = remaining
			kp.keys[i].ResetAt = resetAt
			if remaining == 0 {
				kp.keys[i].Exhausted = true
			}
			return
		}
	}
}

// Size returns the number of keys in the pool.
func (kp *KeyPool) Size() int {
	kp.mu.Lock()
	defer kp.mu.Unlock()
	return len(kp.keys)
}
