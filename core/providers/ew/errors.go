package ew

import (
	"fmt"
	"strings"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// sglangErrorResponse captures the two error shapes SGLang emits on its
// OpenAI-compatible HTTP server (python/sglang/srt/entrypoints/openai/serving_base.py):
//
//	non-streaming (top-level):
//	  {"object":"error","message":"...","type":"BadRequestError","param":null,"code":400}
//
//	streaming (wrapped):
//	  {"error":{"object":"error","message":"...","type":"BadRequestError","param":null,"code":400}}
//
// Code is `interface{}` because SGLang ships an HTTP-status integer there
// (e.g. 400) — we never copy that into bifrostErr.Error.Code (which is a string).
// The actual HTTP status comes from the response header via HandleProviderAPIError.
type sglangErrorResponse struct {
	Object  string              `json:"object,omitempty"`
	Message string              `json:"message,omitempty"`
	Type    string              `json:"type,omitempty"`
	Param   interface{}         `json:"param,omitempty"`
	Code    interface{}         `json:"code,omitempty"`
	Error   *schemas.ErrorField `json:"error,omitempty"`
}

// ParseSGLangError converts an SGLang error response into a BifrostError.
// It is wired in as the customErrorConverter for the EW provider's OpenAI-shaped
// upstream calls so that the non-OpenAI top-level shape is correctly extracted
// (instead of falling back to "provider API error (status N)" via ParseOpenAIError).
func ParseSGLangError(resp *fasthttp.Response, requestType schemas.RequestType, providerName schemas.ModelProvider, model string) *schemas.BifrostError {
	var raw sglangErrorResponse
	bifrostErr := providerUtils.HandleProviderAPIError(resp, &raw)
	if bifrostErr == nil {
		return nil
	}
	if bifrostErr.Error == nil {
		bifrostErr.Error = &schemas.ErrorField{}
	}

	// Streaming-wrapped shape: {"error": {...}}.
	if raw.Error != nil {
		if strings.TrimSpace(raw.Error.Message) != "" {
			bifrostErr.Error.Message = raw.Error.Message
		}
		if raw.Error.Type != nil && strings.TrimSpace(*raw.Error.Type) != "" {
			bifrostErr.Error.Type = raw.Error.Type
			bifrostErr.Type = raw.Error.Type
		}
		if raw.Error.Code != nil && strings.TrimSpace(*raw.Error.Code) != "" {
			bifrostErr.Error.Code = raw.Error.Code
		}
		if raw.Error.Param != nil {
			bifrostErr.Error.Param = raw.Error.Param
		}
		if raw.Error.EventID != nil {
			bifrostErr.Error.EventID = raw.Error.EventID
		}
	}

	// Top-level non-streaming shape — overrides wrapped fields when present.
	if strings.TrimSpace(raw.Message) != "" {
		bifrostErr.Error.Message = raw.Message
	}
	if strings.TrimSpace(raw.Type) != "" {
		errorType := schemas.Ptr(raw.Type)
		bifrostErr.Error.Type = errorType
		bifrostErr.Type = errorType
	}
	if raw.Param != nil {
		bifrostErr.Error.Param = raw.Param
	}

	if strings.TrimSpace(bifrostErr.Error.Message) == "" {
		if bifrostErr.StatusCode != nil {
			bifrostErr.Error.Message = fmt.Sprintf("provider API error (status %d)", *bifrostErr.StatusCode)
		} else {
			bifrostErr.Error.Message = "provider API error"
		}
	}

	bifrostErr.ExtraFields.Provider = providerName
	bifrostErr.ExtraFields.ModelRequested = model
	bifrostErr.ExtraFields.RequestType = requestType

	return bifrostErr
}
