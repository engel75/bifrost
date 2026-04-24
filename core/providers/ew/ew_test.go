package ew_test

import (
	"os"
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/internal/llmtests"
	"github.com/maximhq/bifrost/core/schemas"
)

func TestEW(t *testing.T) {
	t.Parallel()
	baseURL := strings.TrimSpace(os.Getenv("EW_BASE_URL"))
	if baseURL == "" {
		t.Skip("Skipping EW tests because EW_BASE_URL is not set")
	}

	client, ctx, cancel, err := llmtests.SetupTest()
	if err != nil {
		t.Fatalf("Error initializing test setup: %v", err)
	}
	defer cancel()
	defer client.Shutdown()

	chatModel := getEnvWithDefault("EW_CHAT_MODEL", "Qwen/Qwen3-0.6B")
	textModel := getEnvWithDefault("EW_TEXT_MODEL", "Qwen/Qwen3-0.6B")
	reasoningModel := getEnvWithDefault("EW_REASONING_MODEL", "Qwen/Qwen3-0.6B")
	embeddingModel := getEnvWithDefault("EW_EMBEDDING_MODEL", "Qwen3-Embedding-0.6B")
	speechModel := getEnvWithDefault("EW_SPEECH_MODEL", "Qwen/Qwen3-0.6B")
	rerankModel := strings.TrimSpace(os.Getenv("EW_RERANK_MODEL"))

	testConfig := llmtests.ComprehensiveTestConfig{
		Provider:             schemas.EW,
		ChatModel:            chatModel,
		TextModel:            textModel,
		ReasoningModel:       reasoningModel,
		EmbeddingModel:       embeddingModel,
		SpeechSynthesisModel: speechModel,
		RerankModel:          rerankModel,
		Scenarios: llmtests.TestScenarios{
			TextCompletion:        true,
			TextCompletionStream:  true,
			SimpleChat:            true,
			CompletionStream:      true,
			MultiTurnConversation: true,
			ToolCalls:             true,
			ToolCallsStreaming:    true,
			MultipleToolCalls:          true,
			MultipleToolCallsStreaming: true,
			End2EndToolCalling:    true,
			AutomaticFunctionCall: true,
			ImageURL:              false,
			ImageBase64:           false,
			MultipleImages:        false,
			CompleteEnd2End:       true,
			Embedding:             true,
			Rerank:                rerankModel != "",
			ListModels:            true,
			Reasoning:             true,
			SpeechSynthesis:       true,
			SpeechSynthesisStream: true,
			Transcription:         true,
			TranscriptionStream:   false,
			ImageGeneration:       false,
			ImageGenerationStream: false,
			ImageEdit:             false,
			ImageEditStream:       false,
			ImageVariation:        false,
			ImageVariationStream:  false,
		},
	}

	t.Run("EWTests", func(t *testing.T) {
		llmtests.RunAllComprehensiveTests(t, client, ctx, testConfig)
	})
}

func getEnvWithDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}