package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// OpenAIProvider implements the Provider interface for OpenAI's Chat Completions API.
type OpenAIProvider struct {
	client  *http.Client
	baseURL string
}

// NewOpenAIProvider creates a new OpenAI provider.
func NewOpenAIProvider() *OpenAIProvider {
	return &OpenAIProvider{
		client:  &http.Client{},
		baseURL: "https://api.openai.com/v1",
	}
}

func (o *OpenAIProvider) Name() string { return "openai" }

// ---------------------------------------------------------------------------
// Request / Response types for OpenAI Chat Completions
// ---------------------------------------------------------------------------

type openAIRequest struct {
	Model       string           `json:"model"`
	Messages    []openAIMessage  `json:"messages"`
	Temperature float32          `json:"temperature,omitempty"`
	MaxTokens   int32            `json:"max_tokens,omitempty"`
	Stream      bool             `json:"stream,omitempty"`
}

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int32 `json:"prompt_tokens"`
		CompletionTokens int32 `json:"completion_tokens"`
	} `json:"usage"`
}

type openAIStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int32 `json:"prompt_tokens"`
		CompletionTokens int32 `json:"completion_tokens"`
	} `json:"usage,omitempty"`
}

// ---------------------------------------------------------------------------
// Infer — Unary call
// ---------------------------------------------------------------------------

func (o *OpenAIProvider) Infer(ctx context.Context, req Request) (Response, error) {
	body := openAIRequest{
		Model:       req.Model,
		Messages:    []openAIMessage{{Role: "user", Content: req.Prompt}},
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return Response{}, fmt.Errorf("openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return Response{}, fmt.Errorf("openai: create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+req.APIKey)

	httpResp, err := o.client.Do(httpReq)
	if err != nil {
		return Response{}, fmt.Errorf("openai: do request: %w", err)
	}
	defer httpResp.Body.Close()

	if httpResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(httpResp.Body)
		return Response{}, fmt.Errorf("openai: API error %d: %s", httpResp.StatusCode, string(respBody))
	}

	var oaiResp openAIResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&oaiResp); err != nil {
		return Response{}, fmt.Errorf("openai: decode response: %w", err)
	}

	var text string
	if len(oaiResp.Choices) > 0 {
		text = oaiResp.Choices[0].Message.Content
	}

	return Response{
		Text:         text,
		PromptTokens: oaiResp.Usage.PromptTokens,
		OutputTokens: oaiResp.Usage.CompletionTokens,
	}, nil
}

// ---------------------------------------------------------------------------
// InferStream — SSE streaming call
// ---------------------------------------------------------------------------

func (o *OpenAIProvider) InferStream(ctx context.Context, req Request) (<-chan StreamChunk, error) {
	body := openAIRequest{
		Model:       req.Model,
		Messages:    []openAIMessage{{Role: "user", Content: req.Prompt}},
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Stream:      true,
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal stream request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("openai: create stream request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+req.APIKey)

	httpResp, err := o.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: stream request: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		return nil, fmt.Errorf("openai: stream API error %d: %s", httpResp.StatusCode, string(respBody))
	}

	ch := make(chan StreamChunk, 16)

	go func() {
		defer close(ch)
		defer httpResp.Body.Close()

		scanner := bufio.NewScanner(httpResp.Body)
		var totalPromptTokens, totalOutputTokens int32

		for scanner.Scan() {
			select {
			case <-ctx.Done():
				ch <- StreamChunk{Err: ctx.Err()}
				return
			default:
			}

			line := scanner.Text()

			// SSE format: lines starting with "data: "
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")

			// End of stream
			if data == "[DONE]" {
				ch <- StreamChunk{
					Done:         true,
					PromptTokens: totalPromptTokens,
					OutputTokens: totalOutputTokens,
				}
				return
			}

			var chunk openAIStreamChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				ch <- StreamChunk{Err: fmt.Errorf("openai: stream decode: %w", err)}
				return
			}

			// Track usage if reported
			if chunk.Usage != nil {
				totalPromptTokens = chunk.Usage.PromptTokens
				totalOutputTokens = chunk.Usage.CompletionTokens
			}

			var text string
			if len(chunk.Choices) > 0 {
				text = chunk.Choices[0].Delta.Content
			}

			if text != "" {
				ch <- StreamChunk{Text: text}
			}
		}

		if err := scanner.Err(); err != nil {
			ch <- StreamChunk{Err: fmt.Errorf("openai: stream scan: %w", err)}
		}
	}()

	return ch, nil
}
