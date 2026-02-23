package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/abdhe/llm-inference-proxy/pkg/provider"
)

// RedisCache wraps a Redis client for storing and retrieving LLM responses.
type RedisCache struct {
	client *redis.Client
	ttl    time.Duration
}

// NewRedisCache creates a new Redis-backed response cache.
func NewRedisCache(addr, password string, db int, ttl time.Duration) *RedisCache {
	return &RedisCache{
		client: redis.NewClient(&redis.Options{
			Addr:     addr,
			Password: password,
			DB:       db,
		}),
		ttl: ttl,
	}
}

// Get retrieves a cached response by key.
// Returns the response and true if found, or zero value and false if not.
func (r *RedisCache) Get(ctx context.Context, key string) (provider.Response, bool, error) {
	val, err := r.client.Get(ctx, key).Result()
	if err == redis.Nil {
		return provider.Response{}, false, nil
	}
	if err != nil {
		return provider.Response{}, false, fmt.Errorf("redis_cache: get: %w", err)
	}

	var resp provider.Response
	if err := json.Unmarshal([]byte(val), &resp); err != nil {
		return provider.Response{}, false, fmt.Errorf("redis_cache: unmarshal: %w", err)
	}

	return resp, true, nil
}

// Set stores a response in the cache with the configured TTL.
func (r *RedisCache) Set(ctx context.Context, key string, resp provider.Response) error {
	data, err := json.Marshal(resp)
	if err != nil {
		return fmt.Errorf("redis_cache: marshal: %w", err)
	}

	if err := r.client.Set(ctx, key, string(data), r.ttl).Err(); err != nil {
		return fmt.Errorf("redis_cache: set: %w", err)
	}

	return nil
}

// Ping checks the Redis connection.
func (r *RedisCache) Ping(ctx context.Context) error {
	return r.client.Ping(ctx).Err()
}

// Close closes the Redis connection.
func (r *RedisCache) Close() error {
	return r.client.Close()
}
