// Package provider defines the LLM provider interface and shared types.
package provider

import "context"

// Request represents an inference request to an LLM provider.
type Request struct {
	Model       string
	Prompt      string
	Temperature float32
	MaxTokens   int32
	APIKey      string // Injected by the key pool
}

// Response represents a complete inference response.
type Response struct {
	Text         string
	PromptTokens int32
	OutputTokens int32
}

// StreamChunk represents a single chunk in a streaming response.
type StreamChunk struct {
	Text         string
	Done         bool
	PromptTokens int32  // Set on final chunk
	OutputTokens int32  // Set on final chunk
	Err          error  // Non-nil if the stream encountered an error
}

// Provider is the interface that all LLM backends must implement.
type Provider interface {
	// Name returns a human-readable identifier for this provider (e.g. "openai", "gemini").
	Name() string

	// Infer performs a unary (non-streaming) inference call.
	// The context should carry a deadline/timeout.
	Infer(ctx context.Context, req Request) (Response, error)

	// InferStream performs a streaming inference call and returns a channel
	// of StreamChunks. The channel is closed when the stream finishes or
	// the context is cancelled.
	InferStream(ctx context.Context, req Request) (<-chan StreamChunk, error)
}
