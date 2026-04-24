package ew

import (
	"strconv"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

type EWSpeechReference struct {
	AudioPath string `json:"audio_path"`
	Text      string `json:"text"`
}

type EWSpeechRequest struct {
	Model  string `json:"model"`
	Input  string `json:"input"`
	Voice  string `json:"voice,omitempty"`
	Speed  string `json:"speed,omitempty"`
	Format string `json:"response_format,omitempty"`

	RefAudio     string                 `json:"ref_audio,omitempty"`
	RefText      string                 `json:"ref_text,omitempty"`
	References   []EWSpeechReference   `json:"references,omitempty"`
	TaskType     string                 `json:"task_type,omitempty"`
	Language     string                 `json:"language,omitempty"`
	Instructions string                 `json:"instructions,omitempty"`
	StageParams  map[string]interface{} `json:"stage_params,omitempty"`

	ExtraParams map[string]interface{} `json:"-"`
}

func (r *EWSpeechRequest) GetExtraParams() map[string]interface{} {
	return r.ExtraParams
}

func ToEWSpeechRequest(bifrostReq *schemas.BifrostSpeechRequest) *EWSpeechRequest {
	if bifrostReq == nil || bifrostReq.Input == nil || bifrostReq.Input.Input == "" {
		return nil
	}

	ewReq := &EWSpeechRequest{
		Model: bifrostReq.Model,
		Input: bifrostReq.Input.Input,
	}

	if bifrostReq.Params != nil {
		if bifrostReq.Params.VoiceConfig != nil && bifrostReq.Params.VoiceConfig.Voice != nil {
			ewReq.Voice = *bifrostReq.Params.VoiceConfig.Voice
		}
		if bifrostReq.Params.ResponseFormat != "" {
			ewReq.Format = bifrostReq.Params.ResponseFormat
		}
		if bifrostReq.Params.Speed != nil {
			ewReq.Speed = strconv.FormatFloat(*bifrostReq.Params.Speed, 'f', -1, 64)
		}
		ewReq.Instructions = bifrostReq.Params.Instructions

		if bifrostReq.Params.LanguageCode != nil {
			ewReq.Language = *bifrostReq.Params.LanguageCode
		}

		if bifrostReq.Params.ExtraParams != nil {
			ewReq.ExtraParams = bifrostReq.Params.ExtraParams

			if v, ok := extractString(bifrostReq.Params.ExtraParams, "ref_audio"); ok {
				ewReq.RefAudio = v
			}
			if v, ok := extractString(bifrostReq.Params.ExtraParams, "ref_text"); ok {
				ewReq.RefText = v
			}
			if v, ok := extractString(bifrostReq.Params.ExtraParams, "task_type"); ok {
				ewReq.TaskType = v
			}
			if v, ok := extractString(bifrostReq.Params.ExtraParams, "language"); ok {
				ewReq.Language = v
			}
			if v, ok := extractString(bifrostReq.Params.ExtraParams, "instructions"); ok {
				ewReq.Instructions = v
			}
			if refs, ok := extractReferences(bifrostReq.Params.ExtraParams, "references"); ok {
				ewReq.References = refs
			}
			if sp, ok := extractMap(bifrostReq.Params.ExtraParams, "stage_params"); ok {
				ewReq.StageParams = sp
			}
		}
	}

	return ewReq
}

func extractString(params map[string]interface{}, key string) (string, bool) {
	v, ok := params[key]
	if !ok || v == nil {
		return "", false
	}
	if s, ok := v.(string); ok {
		return s, true
	}
	return "", false
}

func extractReferences(params map[string]interface{}, key string) ([]EWSpeechReference, bool) {
	v, ok := params[key]
	if !ok || v == nil {
		return nil, false
	}
	items, ok := v.([]interface{})
	if !ok {
		return nil, false
	}
	refs := make([]EWSpeechReference, 0, len(items))
	for _, item := range items {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		ref := EWSpeechReference{}
		if ap, ok := m["audio_path"].(string); ok {
			ref.AudioPath = ap
		}
		if t, ok := m["text"].(string); ok {
			ref.Text = t
		}
		refs = append(refs, ref)
	}
	return refs, true
}

func extractMap(params map[string]interface{}, key string) (map[string]interface{}, bool) {
	v, ok := params[key]
	if !ok || v == nil {
		return nil, false
	}
	m, ok := v.(map[string]interface{})
	return m, ok
}