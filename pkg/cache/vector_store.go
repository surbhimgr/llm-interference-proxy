package cache

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/google/uuid"
)

// VectorStore provides similarity search over embeddings using Qdrant.
type VectorStore struct {
	client     *http.Client
	baseURL    string
	collection string
}

// SearchResult holds the result of a similarity search.
type SearchResult struct {
	ID    string
	Score float32
	Found bool
}

// NewVectorStore creates a new Qdrant-backed vector store.
func NewVectorStore(qdrantURL, collection string) *VectorStore {
	return &VectorStore{
		client:     &http.Client{},
		baseURL:    qdrantURL,
		collection: collection,
	}
}

// ---------------------------------------------------------------------------
// Qdrant API types
// ---------------------------------------------------------------------------

type qdrantSearchRequest struct {
	Vector      []float32 `json:"vector"`
	Limit       int       `json:"limit"`
	ScoreThresh float32   `json:"score_threshold"`
	WithPayload bool      `json:"with_payload"`
}

type qdrantSearchResponse struct {
	Result []struct {
		ID      string  `json:"id"`
		Score   float32 `json:"score"`
		Payload map[string]interface{} `json:"payload"`
	} `json:"result"`
}

type qdrantUpsertRequest struct {
	Points []qdrantPoint `json:"points"`
}

type qdrantPoint struct {
	ID      string            `json:"id"`
	Vector  []float32         `json:"vector"`
	Payload map[string]string `json:"payload"`
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Search finds the nearest neighbor in the vector store above the given threshold.
func (v *VectorStore) Search(ctx context.Context, vector []float32, threshold float32) (SearchResult, error) {
	body := qdrantSearchRequest{
		Vector:      vector,
		Limit:       1,
		ScoreThresh: threshold,
		WithPayload: true,
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return SearchResult{}, fmt.Errorf("vector_store: marshal search: %w", err)
	}

	url := fmt.Sprintf("%s/collections/%s/points/search", v.baseURL, v.collection)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return SearchResult{}, fmt.Errorf("vector_store: create search request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := v.client.Do(req)
	if err != nil {
		return SearchResult{}, fmt.Errorf("vector_store: search request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return SearchResult{}, fmt.Errorf("vector_store: search error %d: %s", resp.StatusCode, string(respBody))
	}

	var searchResp qdrantSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return SearchResult{}, fmt.Errorf("vector_store: decode search: %w", err)
	}

	if len(searchResp.Result) == 0 {
		return SearchResult{Found: false}, nil
	}

	top := searchResp.Result[0]
	return SearchResult{
		ID:    top.ID,
		Score: top.Score,
		Found: true,
	}, nil
}

// Upsert stores a vector with the given cache key as payload.
func (v *VectorStore) Upsert(ctx context.Context, cacheKey string, vector []float32) error {
	body := qdrantUpsertRequest{
		Points: []qdrantPoint{
			{
				ID:      uuid.New().String(),
				Vector:  vector,
				Payload: map[string]string{"cache_key": cacheKey},
			},
		},
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("vector_store: marshal upsert: %w", err)
	}

	url := fmt.Sprintf("%s/collections/%s/points", v.baseURL, v.collection)
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, bytes.NewReader(jsonBody))
	if err != nil {
		return fmt.Errorf("vector_store: create upsert request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := v.client.Do(req)
	if err != nil {
		return fmt.Errorf("vector_store: upsert request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("vector_store: upsert error %d: %s", resp.StatusCode, string(respBody))
	}

	return nil
}
