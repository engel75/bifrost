package ew

import (
	"fmt"
	"sort"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// ToEWRerankRequest converts a Bifrost rerank request to EW format.
func ToEWRerankRequest(bifrostReq *schemas.BifrostRerankRequest) *EWRerankRequest {
	if bifrostReq == nil {
		return nil
	}

	ewReq := &EWRerankRequest{
		Model:     bifrostReq.Model,
		Query:     bifrostReq.Query,
		Documents: make([]string, len(bifrostReq.Documents)),
	}

	for i, doc := range bifrostReq.Documents {
		ewReq.Documents[i] = doc.Text
	}

	if bifrostReq.Params != nil {
		ewReq.TopN = bifrostReq.Params.TopN
		ewReq.MaxTokensPerDoc = bifrostReq.Params.MaxTokensPerDoc
		ewReq.Priority = bifrostReq.Params.Priority
		ewReq.ExtraParams = bifrostReq.Params.ExtraParams
	}

	return ewReq
}

// ToBifrostRerankResponse converts an EW rerank response payload to Bifrost format.
func ToBifrostRerankResponse(payload map[string]interface{}, documents []schemas.RerankDocument, returnDocuments bool) (*schemas.BifrostRerankResponse, error) {
	if payload == nil {
		return nil, fmt.Errorf("ew rerank response is nil")
	}

	response := &schemas.BifrostRerankResponse{}

	if id, ok := schemas.SafeExtractString(payload["id"]); ok {
		response.ID = id
	}
	if model, ok := schemas.SafeExtractString(payload["model"]); ok {
		response.Model = model
	}
	if usage, ok := parseEWUsage(payload["usage"]); ok {
		response.Usage = usage
	}

	resultsRaw := payload["results"]
	if resultsRaw == nil {
		return nil, fmt.Errorf("invalid ew rerank response: missing results")
	}

	resultItems, ok := resultsRaw.([]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid ew rerank response: results must be an array")
	}

	seenIndices := make(map[int]struct{}, len(resultItems))
	response.Results = make([]schemas.RerankResult, 0, len(resultItems))

	for _, item := range resultItems {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid ew rerank response: result item must be an object")
		}

		index, ok := schemas.SafeExtractInt(itemMap["index"])
		if !ok {
			return nil, fmt.Errorf("invalid ew rerank response: result index is required")
		}
		if index < 0 || index >= len(documents) {
			return nil, fmt.Errorf("invalid ew rerank response: result index %d out of range", index)
		}
		if _, exists := seenIndices[index]; exists {
			return nil, fmt.Errorf("invalid ew rerank response: duplicate index %d", index)
		}
		seenIndices[index] = struct{}{}

		relevanceScore, ok := schemas.SafeExtractFloat64(itemMap["relevance_score"])
		if !ok {
			relevanceScore, ok = schemas.SafeExtractFloat64(itemMap["score"])
		}
		if !ok {
			return nil, fmt.Errorf("invalid ew rerank response: relevance_score/score is required")
		}

		result := schemas.RerankResult{
			Index:          index,
			RelevanceScore: relevanceScore,
		}

		if returnDocuments {
			doc := documents[index]
			result.Document = &doc
		}

		response.Results = append(response.Results, result)
	}

	sort.SliceStable(response.Results, func(i, j int) bool {
		if response.Results[i].RelevanceScore == response.Results[j].RelevanceScore {
			return response.Results[i].Index < response.Results[j].Index
		}
		return response.Results[i].RelevanceScore > response.Results[j].RelevanceScore
	})

	return response, nil
}

func parseEWUsage(rawUsage interface{}) (*schemas.BifrostLLMUsage, bool) {
	usageMap, ok := rawUsage.(map[string]interface{})
	if !ok {
		return nil, false
	}

	promptTokens := 0
	if _, hasPromptTokens := usageMap["prompt_tokens"]; hasPromptTokens {
		promptTokens, _ = schemas.SafeExtractInt(usageMap["prompt_tokens"])
	} else {
		promptTokens, _ = schemas.SafeExtractInt(usageMap["input_tokens"])
	}

	completionTokens := 0
	if _, hasCompletionTokens := usageMap["completion_tokens"]; hasCompletionTokens {
		completionTokens, _ = schemas.SafeExtractInt(usageMap["completion_tokens"])
	} else {
		completionTokens, _ = schemas.SafeExtractInt(usageMap["output_tokens"])
	}

	totalTokens, ok := schemas.SafeExtractInt(usageMap["total_tokens"])
	if !ok {
		totalTokens = promptTokens + completionTokens
	}
	if promptTokens == 0 && completionTokens == 0 && totalTokens == 0 {
		return nil, false
	}

	return &schemas.BifrostLLMUsage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      totalTokens,
	}, true
}