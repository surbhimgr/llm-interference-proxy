// Package cache implements semantic caching with vector similarity search.
package cache

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Embedder generates vector embeddings for text queries.
type Embedder struct {
	client  *http.Client
	baseURL string
	model   string
	apiKey  string
}

// NewEmbedder creates a new Embedder backed by OpenAI's embedding API.
func NewEmbedder(apiKey string) *Embedder {
	return &Embedder{
		client:  &http.Client{},
		baseURL: "https://api.openai.com/v1",
		model:   "text-embedding-3-small",
		apiKey:  apiKey,
	}
}

// embeddingRequest is the OpenAI embedding API request body.
type embeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

// embeddingResponse is the OpenAI embedding API response body.
type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// Embed generates a vector embedding for the given text.
func (e *Embedder) Embed(ctx context.Context, text string) ([]float32, error) {
	body := embeddingRequest{
		Input: text,
		Model: e.model,
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("embedder: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/embeddings", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("embedder: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embedder: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedder: API error %d: %s", resp.StatusCode, string(respBody))
	}

	var embResp embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
		return nil, fmt.Errorf("embedder: decode: %w", err)
	}

	if len(embResp.Data) == 0 {
		return nil, fmt.Errorf("embedder: empty embedding response")
	}

	return embResp.Data[0].Embedding, nil
}
