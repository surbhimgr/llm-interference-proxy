package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// GeminiProvider implements the Provider interface for Google's Gemini API.
type GeminiProvider struct {
	client  *http.Client
	baseURL string
}

// NewGeminiProvider creates a new Gemini provider.
func NewGeminiProvider() *GeminiProvider {
	return &GeminiProvider{
		client:  &http.Client{},
		baseURL: "https://generativelanguage.googleapis.com/v1beta",
	}
}

func (g *GeminiProvider) Name() string { return "gemini" }

// geminiRequest is the Gemini API request body.
type geminiRequest struct {
	Contents         []geminiContent   `json:"contents"`
	GenerationConfig *geminiGenConfig  `json:"generationConfig,omitempty"`
}

type geminiContent struct {
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiGenConfig struct {
	Temperature  float32 `json:"temperature,omitempty"`
	MaxOutputTokens int32 `json:"maxOutputTokens,omitempty"`
}

// geminiResponse is the Gemini API response body.
type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int32 `json:"promptTokenCount"`
		CandidatesTokenCount int32 `json:"candidatesTokenCount"`
	} `json:"usageMetadata"`
}

// Infer performs a unary inference call to the Gemini API.
func (g *GeminiProvider) Infer(ctx context.Context, req Request) (Response, error) {
	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", g.baseURL, req.Model, req.APIKey)

	body := geminiRequest{
		Contents: []geminiContent{
			{Parts: []geminiPart{{Text: req.Prompt}}},
		},
		GenerationConfig: &geminiGenConfig{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxTokens,
		},
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return Response{}, fmt.Errorf("gemini: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return Response{}, fmt.Errorf("gemini: create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := g.client.Do(httpReq)
	if err != nil {
		return Response{}, fmt.Errorf("gemini: do request: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(httpResp.Body)
		return Response{}, fmt.Errorf("gemini: API error %d: %s", httpResp.StatusCode, string(respBody))
	}

	var gemResp geminiResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&gemResp); err != nil {
		return Response{}, fmt.Errorf("gemini: decode response: %w", err)
	}

	var text string
	if len(gemResp.Candidates) > 0 && len(gemResp.Candidates[0].Content.Parts) > 0 {
		text = gemResp.Candidates[0].Content.Parts[0].Text
	}

	return Response{
		Text:         text,
		PromptTokens: gemResp.UsageMetadata.PromptTokenCount,
		OutputTokens: gemResp.UsageMetadata.CandidatesTokenCount,
	}, nil
}

// InferStream performs a streaming inference call to the Gemini API.
func (g *GeminiProvider) InferStream(ctx context.Context, req Request) (<-chan StreamChunk, error) {
	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?key=%s&alt=sse", g.baseURL, req.Model, req.APIKey)

	body := geminiRequest{
		Contents: []geminiContent{
			{Parts: []geminiPart{{Text: req.Prompt}}},
		},
		GenerationConfig: &geminiGenConfig{
			Temperature:     req.Temperature,
			MaxOutputTokens: req.MaxTokens,
		},
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal stream request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("gemini: create stream request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := g.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: stream request: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		return nil, fmt.Errorf("gemini: stream API error %d: %s", httpResp.StatusCode, string(respBody))
	}

	ch := make(chan StreamChunk, 16)

	go func() {
		defer close(ch)
		defer httpResp.Body.Close()

		decoder := json.NewDecoder(httpResp.Body)
		for {
			select {
			case <-ctx.Done():
				ch <- StreamChunk{Err: ctx.Err()}
				return
			default:
			}

			var gemResp geminiResponse
			if err := decoder.Decode(&gemResp); err != nil {
				if err != io.EOF {
					ch <- StreamChunk{Err: fmt.Errorf("gemini: stream decode: %w", err)}
				}
				// Send final chunk
				ch <- StreamChunk{Done: true}
				return
			}

			var text string
			if len(gemResp.Candidates) > 0 && len(gemResp.Candidates[0].Content.Parts) > 0 {
				text = gemResp.Candidates[0].Content.Parts[0].Text
			}

			ch <- StreamChunk{
				Text:         text,
				PromptTokens: gemResp.UsageMetadata.PromptTokenCount,
				OutputTokens: gemResp.UsageMetadata.CandidatesTokenCount,
			}
		}
	}()

	return ch, nil
}
