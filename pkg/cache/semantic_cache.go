package cache

import (
	"context"
	"crypto/sha256"
	"fmt"
	"log"

	"github.com/abdhe/llm-inference-proxy/pkg/provider"
)

// SemanticCache orchestrates the embed → search → hit/miss caching flow.
type SemanticCache struct {
	embedder    *Embedder
	vectorStore *VectorStore
	redisCache  *RedisCache
	threshold   float32 // Similarity threshold (e.g. 0.95)
}

// NewSemanticCache creates a new semantic cache.
func NewSemanticCache(embedder *Embedder, vectorStore *VectorStore, redisCache *RedisCache, threshold float32) *SemanticCache {
	return &SemanticCache{
		embedder:    embedder,
		vectorStore: vectorStore,
		redisCache:  redisCache,
		threshold:   threshold,
	}
}

// CacheResult holds the result of a cache lookup.
type CacheResult struct {
	Response provider.Response
	Hit      bool
}

// Lookup checks the semantic cache for a similar query.
// Flow:
//  1. Generate an embedding for the prompt.
//  2. Search the vector store for a neighbor above the similarity threshold.
//  3. If found, retrieve the cached response from Redis.
//  4. If not found, return a cache miss.
func (sc *SemanticCache) Lookup(ctx context.Context, prompt string) (CacheResult, error) {
	// Step 1: Embed the query
	vector, err := sc.embedder.Embed(ctx, prompt)
	if err != nil {
		// Log but don't fail — treat as cache miss
		log.Printf("[semantic_cache] embedding error (treating as miss): %v", err)
		return CacheResult{Hit: false}, nil
	}

	// Step 2: Search for similar vectors
	result, err := sc.vectorStore.Search(ctx, vector, sc.threshold)
	if err != nil {
		log.Printf("[semantic_cache] vector search error (treating as miss): %v", err)
		return CacheResult{Hit: false}, nil
	}

	if !result.Found {
		return CacheResult{Hit: false}, nil
	}

	// Step 3: Retrieve from Redis using the cache key from payload
	cacheKey := result.ID
	resp, found, err := sc.redisCache.Get(ctx, cacheKey)
	if err != nil {
		log.Printf("[semantic_cache] redis get error (treating as miss): %v", err)
		return CacheResult{Hit: false}, nil
	}

	if !found {
		// Vector was in Qdrant but Redis entry expired — treat as miss
		return CacheResult{Hit: false}, nil
	}

	return CacheResult{
		Response: resp,
		Hit:      true,
	}, nil
}

// Store caches a prompt-response pair.
// Flow:
//  1. Generate an embedding for the prompt.
//  2. Create a deterministic cache key from the prompt.
//  3. Store the response in Redis.
//  4. Upsert the embedding into the vector store with the cache key.
func (sc *SemanticCache) Store(ctx context.Context, prompt string, resp provider.Response) {
	// Step 1: Embed
	vector, err := sc.embedder.Embed(ctx, prompt)
	if err != nil {
		log.Printf("[semantic_cache] store embedding error: %v", err)
		return
	}

	// Step 2: Deterministic cache key
	cacheKey := cacheKeyFromPrompt(prompt)

	// Step 3: Store in Redis
	if err := sc.redisCache.Set(ctx, cacheKey, resp); err != nil {
		log.Printf("[semantic_cache] redis set error: %v", err)
		return
	}

	// Step 4: Upsert vector
	if err := sc.vectorStore.Upsert(ctx, cacheKey, vector); err != nil {
		log.Printf("[semantic_cache] vector upsert error: %v", err)
	}
}

// cacheKeyFromPrompt generates a deterministic cache key for a prompt.
func cacheKeyFromPrompt(prompt string) string {
	hash := sha256.Sum256([]byte(prompt))
	return fmt.Sprintf("llm_cache:%x", hash[:16])
}
