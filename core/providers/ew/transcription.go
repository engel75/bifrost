package ew

import (
	"github.com/bytedance/sonic"
	schemas "github.com/maximhq/bifrost/core/schemas"
)

func parseEWTranscriptionStreamChunk(jsonData []byte) (*schemas.BifrostTranscriptionStreamResponse, bool) {
	var chunk ewTranscriptionStreamChunk
	response := &schemas.BifrostTranscriptionStreamResponse{}
	if err := sonic.Unmarshal(jsonData, &chunk); err != nil {
		return nil, false
	}
	if chunk.Usage != nil {
		return &schemas.BifrostTranscriptionStreamResponse{
			Type:  schemas.TranscriptionStreamResponseTypeDone,
			Usage: chunk.Usage,
		}, true
	}
	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta.Content == nil {
		return nil, false
	}
	if len(chunk.Choices) > 0 {
		reason := chunk.Choices[0].FinishReason
		if reason == nil && chunk.Choices[0].StopReason != nil {
			reason = chunk.Choices[0].StopReason
		}
		if reason != nil && *reason == "stop" {
			response.Text = *chunk.Choices[0].Delta.Content
			response.Type = schemas.TranscriptionStreamResponseTypeDone
		} else {
			response.Type = schemas.TranscriptionStreamResponseTypeDelta
		}
		response.Delta = chunk.Choices[0].Delta.Content
	}
	return response, true
}
