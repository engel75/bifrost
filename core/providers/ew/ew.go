// Package ew implements the EW LLM provider (OpenAI-compatible).
package ew

import (
	"bytes"
	"context"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"time"

	"github.com/maximhq/bifrost/core/providers/openai"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// EWProvider implements the Provider interface for EW's OpenAI-compatible API.
type EWProvider struct {
	logger              schemas.Logger        // Logger for provider operations
	client              *fasthttp.Client      // HTTP client for API requests
	networkConfig       schemas.NetworkConfig // Network configuration including extra headers
	sendBackRawRequest  bool                  // Whether to include raw request in BifrostResponse
	sendBackRawResponse bool                  // Whether to include raw response in BifrostResponse
}

// NewEWProvider creates a new EW provider instance.
func NewEWProvider(config *schemas.ProviderConfig, logger schemas.Logger) (*EWProvider, error) {
	config.CheckAndSetDefaults()

	requestTimeout := time.Second * time.Duration(config.NetworkConfig.DefaultRequestTimeoutInSeconds)
	client := &fasthttp.Client{
		ReadTimeout:         requestTimeout,
		WriteTimeout:        requestTimeout,
		MaxConnsPerHost:     config.NetworkConfig.MaxConnsPerHost,
		MaxIdleConnDuration: 30 * time.Second,
		MaxConnWaitTimeout:  requestTimeout,
		MaxConnDuration:     time.Second * time.Duration(schemas.DefaultMaxConnDurationInSeconds),
		ConnPoolStrategy:    fasthttp.FIFO,
	}

	client = providerUtils.ConfigureProxy(client, config.ProxyConfig, logger)
	client = providerUtils.ConfigureDialer(client)
	client = providerUtils.ConfigureTLS(client, config.NetworkConfig, logger)
	config.NetworkConfig.BaseURL = strings.TrimRight(config.NetworkConfig.BaseURL, "/")

	// BaseURL is optional when keys have ew_key_config with per-key URLs
	return &EWProvider{
		logger:              logger,
		client:              client,
		networkConfig:       config.NetworkConfig,
		sendBackRawRequest:  config.SendBackRawRequest,
		sendBackRawResponse: config.SendBackRawResponse,
	}, nil
}

// GetProviderKey returns the provider identifier for EW.
func (provider *EWProvider) GetProviderKey() schemas.ModelProvider {
	return schemas.EW
}

// getBaseURL resolves the base URL for a request from the per-key ew_key_config.
// Each EW key must have its own URL configured — there is no provider-level fallback.
func (provider *EWProvider) getBaseURL(key schemas.Key) string {
	if key.EWKeyConfig != nil && key.EWKeyConfig.URL.GetValue() != "" {
		return strings.TrimRight(key.EWKeyConfig.URL.GetValue(), "/")
	}
	return ""
}

// baseURLOrError returns the resolved base URL or a BifrostError when none is configured.
func (provider *EWProvider) baseURLOrError(key schemas.Key) (string, *schemas.BifrostError) {
	u := provider.getBaseURL(key)
	if u == "" {
		return "", providerUtils.NewBifrostOperationError(
			"no base URL configured: set ew_key_config.url on the key",
			nil,
			provider.GetProviderKey(),
		)
	}
	return u, nil
}

// listModelsByKey performs a list models request for a single EW key,
// resolving the per-key URL so each backend is queried individually.
func (provider *EWProvider) listModelsByKey(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostListModelsRequest) (*schemas.BifrostListModelsResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	url := baseURL + providerUtils.GetPathFromContext(ctx, "/v1/models")
	return openai.ListModelsByKey(
		ctx,
		provider.client,
		url,
		key,
		request.Unfiltered,
		provider.networkConfig.ExtraHeaders,
		provider.GetProviderKey(),
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
	)
}

// ListModels performs a list models request to EW's API.
// Requests are made concurrently per key so that each backend is queried
// with its own URL (from ew_key_config).
func (provider *EWProvider) ListModels(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostListModelsRequest) (*schemas.BifrostListModelsResponse, *schemas.BifrostError) {
	return providerUtils.HandleMultipleListModelsRequests(
		ctx,
		keys,
		request,
		provider.listModelsByKey,
	)
}

// TextCompletion performs a text completion request to EW's API.
func (provider *EWProvider) TextCompletion(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostTextCompletionRequest) (*schemas.BifrostTextCompletionResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	return openai.HandleOpenAITextCompletionRequest(
		ctx,
		provider.client,
		baseURL+providerUtils.GetPathFromContext(ctx, "/v1/completions"),
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		provider.GetProviderKey(),
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		HandleEWResponse,
		nil,
		provider.logger,
	)
}

// TextCompletionStream performs a streaming text completion request to EW's API.
func (provider *EWProvider) TextCompletionStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostTextCompletionRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	var authHeader map[string]string
	if key.Value.GetValue() != "" {
		authHeader = map[string]string{"Authorization": "Bearer " + key.Value.GetValue()}
	}
	return openai.HandleOpenAITextCompletionStreaming(
		ctx,
		provider.client,
		baseURL+providerUtils.GetPathFromContext(ctx, "/v1/completions"),
		request,
		authHeader,
		provider.networkConfig.ExtraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		provider.GetProviderKey(),
		nil,
		postHookRunner,
		HandleEWResponse,
		nil,
		provider.logger,
	)
}

// ChatCompletion performs a chat completion request to EW's API.
func (provider *EWProvider) ChatCompletion(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostChatRequest) (*schemas.BifrostChatResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	return openai.HandleOpenAIChatCompletionRequest(
		ctx,
		provider.client,
		baseURL+providerUtils.GetPathFromContext(ctx, "/v1/chat/completions"),
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		provider.GetProviderKey(),
		HandleEWResponse,
		nil,
		provider.logger,
	)
}

// ChatCompletionStream performs a streaming chat completion request to EW's API.
func (provider *EWProvider) ChatCompletionStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostChatRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	var authHeader map[string]string
	if key.Value.GetValue() != "" {
		authHeader = map[string]string{"Authorization": "Bearer " + key.Value.GetValue()}
	}
	return openai.HandleOpenAIChatCompletionStreaming(
		ctx,
		provider.client,
		baseURL+providerUtils.GetPathFromContext(ctx, "/v1/chat/completions"),
		request,
		authHeader,
		provider.networkConfig.ExtraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		provider.GetProviderKey(),
		postHookRunner,
		nil,
		HandleEWResponse,
		nil,
		nil,
		nil,
		provider.logger,
	)
}

// Embedding performs an embedding request to EW's API.
func (provider *EWProvider) Embedding(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostEmbeddingRequest) (*schemas.BifrostEmbeddingResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	return openai.HandleOpenAIEmbeddingRequest(
		ctx,
		provider.client,
		baseURL+providerUtils.GetPathFromContext(ctx, "/v1/embeddings"),
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		provider.GetProviderKey(),
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		HandleEWResponse,
		provider.logger,
	)
}

// Responses performs a responses request to EW's API (via chat completion).
func (provider *EWProvider) Responses(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostResponsesRequest) (*schemas.BifrostResponsesResponse, *schemas.BifrostError) {
	chatResponse, err := provider.ChatCompletion(ctx, key, request.ToChatRequest())
	if err != nil {
		return nil, err
	}
	response := chatResponse.ToBifrostResponsesResponse()
	response.ExtraFields.RequestType = schemas.ResponsesRequest
	response.ExtraFields.Provider = provider.GetProviderKey()
	response.ExtraFields.ModelRequested = request.Model
	return response, nil
}

// ResponsesStream performs a streaming responses request to EW's API (via chat completion stream).
func (provider *EWProvider) ResponsesStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostResponsesRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	ctx.SetValue(schemas.BifrostContextKeyIsResponsesToChatCompletionFallback, true)
	return provider.ChatCompletionStream(
		ctx,
		postHookRunner,
		key,
		request.ToChatRequest(),
	)
}

func (provider *EWProvider) Speech(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostSpeechRequest) (*schemas.BifrostSpeechResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	return openai.HandleOpenAISpeechRequest(
		ctx,
		provider.client,
		baseURL+providerUtils.GetPathFromContext(ctx, "/v1/audio/speech"),
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		provider.GetProviderKey(),
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		nil,
		provider.logger,
	)
}

func isRerankFallbackStatus(statusCode int) bool {
	// EW deployments may return 501 for unimplemented routes.
	// We fallback on 501 in addition to 404/405 for compatibility.
	return statusCode == fasthttp.StatusNotFound ||
		statusCode == fasthttp.StatusMethodNotAllowed ||
		statusCode == fasthttp.StatusNotImplemented
}

func (provider *EWProvider) callEWRerankEndpoint(
	ctx *schemas.BifrostContext,
	key schemas.Key,
	request *schemas.BifrostRerankRequest,
	endpointPath string,
	jsonData []byte,
) (map[string]interface{}, interface{}, interface{}, []byte, int, time.Duration, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, nil, nil, nil, 0, 0, bifrostErr
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)

	req.SetRequestURI(baseURL + endpointPath)
	req.Header.SetMethod(http.MethodPost)
	req.Header.SetContentType("application/json")

	if key.Value.GetValue() != "" {
		req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	}
	if !providerUtils.ApplyLargePayloadRequestBodyWithModelNormalization(ctx, req, schemas.EW) {
		req.SetBody(jsonData)
	}

	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, nil, nil, nil, 0, latency, bifrostErr
	}

	statusCode := resp.StatusCode()
	if statusCode != fasthttp.StatusOK {
		return nil, nil, nil, nil, statusCode, latency, openai.ParseOpenAIError(resp, schemas.RerankRequest, provider.GetProviderKey(), request.Model)
	}

	body, err := providerUtils.CheckAndDecodeBody(resp)
	if err != nil {
		return nil, nil, nil, nil, statusCode, latency, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseDecode, err, provider.GetProviderKey())
	}

	sendBackRawRequest := providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest)
	sendBackRawResponse := providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse)

	responsePayload := make(map[string]interface{})
	rawRequest, rawResponse, bifrostErr := HandleEWResponse(body, &responsePayload, jsonData, sendBackRawRequest, sendBackRawResponse)
	if bifrostErr != nil {
		return nil, nil, nil, body, statusCode, latency, bifrostErr
	}

	return responsePayload, rawRequest, rawResponse, body, statusCode, latency, nil
}

// Rerank performs a rerank request to EW's API.
func (provider *EWProvider) Rerank(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostRerankRequest) (*schemas.BifrostRerankResponse, *schemas.BifrostError) {
	providerName := provider.GetProviderKey()

	jsonData, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
		ctx,
		request,
		func() (providerUtils.RequestBodyWithExtraParams, error) {
			return ToEWRerankRequest(request), nil
		},
		providerName,
	)
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	resolvedPath := providerUtils.GetPathFromContext(ctx, "")
	hasPathOverride := resolvedPath != ""
	if !hasPathOverride {
		resolvedPath = "/v1/rerank"
	} else if !strings.HasPrefix(resolvedPath, "/") {
		resolvedPath = "/" + resolvedPath
	}

	responsePayload, rawRequest, rawResponse, responseBody, statusCode, latency, bifrostErr := provider.callEWRerankEndpoint(ctx, key, request, resolvedPath, jsonData)
	if bifrostErr != nil && !hasPathOverride && isRerankFallbackStatus(statusCode) {
		var fallbackLatency time.Duration
		responsePayload, rawRequest, rawResponse, responseBody, statusCode, fallbackLatency, bifrostErr = provider.callEWRerankEndpoint(ctx, key, request, "/rerank", jsonData)
		latency += fallbackLatency
	}
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, responseBody, provider.sendBackRawRequest, provider.sendBackRawResponse)
	}

	returnDocuments := request.Params != nil && request.Params.ReturnDocuments != nil && *request.Params.ReturnDocuments
	bifrostResponse, err := ToBifrostRerankResponse(responsePayload, request.Documents, returnDocuments)
	if err != nil {
		return nil, providerUtils.EnrichError(
			ctx,
			providerUtils.NewBifrostOperationError("error converting rerank response", err, providerName),
			jsonData,
			responseBody,
			provider.sendBackRawRequest,
			provider.sendBackRawResponse,
		)
	}

	// Keep requested model as the canonical model in Bifrost response.
	bifrostResponse.Model = request.Model
	bifrostResponse.ExtraFields.Provider = providerName
	bifrostResponse.ExtraFields.ModelRequested = request.Model
	bifrostResponse.ExtraFields.RequestType = schemas.RerankRequest
	bifrostResponse.ExtraFields.Latency = latency.Milliseconds()

	if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
		bifrostResponse.ExtraFields.RawRequest = rawRequest
	}
	if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
		bifrostResponse.ExtraFields.RawResponse = rawResponse
	}

	return bifrostResponse, nil
}

// OCR is not supported by the EW provider.
func (provider *EWProvider) OCR(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostOCRRequest) (*schemas.BifrostOCRResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.OCRRequest, provider.GetProviderKey())
}

// SpeechStream is not supported by the EW provider.
func (provider *EWProvider) SpeechStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostSpeechRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.SpeechStreamRequest, provider.GetProviderKey())
}

// Transcription performs a transcription request to EW's API.
func (provider *EWProvider) Transcription(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostTranscriptionRequest) (*schemas.BifrostTranscriptionResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	return openai.HandleOpenAITranscriptionRequest(
		ctx,
		provider.client,
		baseURL+providerUtils.GetPathFromContext(ctx, "/v1/audio/transcriptions"),
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		provider.GetProviderKey(),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		HandleEWResponse,
		provider.logger,
	)
}

// TranscriptionStream performs a streaming transcription request to EW's API.
func (provider *EWProvider) TranscriptionStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostTranscriptionRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	{
		logger := provider.logger
		providerName := provider.GetProviderKey()
		// Use centralized converter
		reqBody := openai.ToOpenAITranscriptionRequest(request)
		if reqBody == nil {
			return nil, providerUtils.NewBifrostOperationError("transcription input is not provided", nil, providerName)
		}
		reqBody.Stream = schemas.Ptr(true)

		// Create multipart form
		var body bytes.Buffer
		writer := multipart.NewWriter(&body)

		if bifrostErr := openai.ParseTranscriptionFormDataBodyFromRequest(writer, reqBody, providerName); bifrostErr != nil {
			return nil, bifrostErr
		}

		// Prepare OpenAI headers
		headers := map[string]string{
			"Content-Type":  writer.FormDataContentType(),
			"Accept":        "text/event-stream",
			"Cache-Control": "no-cache",
		}
		if key.Value.GetValue() != "" {
			headers["Authorization"] = "Bearer " + key.Value.GetValue()
		}

		// Create HTTP request for streaming
		req := fasthttp.AcquireRequest()
		resp := fasthttp.AcquireResponse()
		resp.StreamBody = true
		defer fasthttp.ReleaseRequest(req)

		// Set any extra headers from network config
		providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)

		req.Header.SetMethod(http.MethodPost)
		req.SetRequestURI(baseURL + providerUtils.GetPathFromContext(ctx, "/v1/audio/transcriptions"))

		// Set headers
		for key, value := range headers {
			req.Header.Set(key, value)
		}

		req.SetBody(body.Bytes())

		// Make the request
		err := provider.client.Do(req, resp)
		if err != nil {
			defer providerUtils.ReleaseStreamingResponse(resp)
			if errors.Is(err, context.Canceled) {
				return nil, &schemas.BifrostError{
					IsBifrostError: false,
					Error: &schemas.ErrorField{
						Type:    schemas.Ptr(schemas.RequestCancelled),
						Message: schemas.ErrRequestCancelled,
						Error:   err,
					},
				}
			}
			if errors.Is(err, fasthttp.ErrTimeout) || errors.Is(err, context.DeadlineExceeded) {
				return nil, providerUtils.NewBifrostTimeoutError(schemas.ErrProviderRequestTimedOut, err, providerName)
			}
			return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderDoRequest, err, providerName)
		}

		// Store provider response headers in context before status check so error responses also forward them
		ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerUtils.ExtractProviderResponseHeaders(resp))

		// Check for HTTP errors
		if resp.StatusCode() != fasthttp.StatusOK {
			defer providerUtils.ReleaseStreamingResponse(resp)
			return nil, openai.ParseOpenAIError(resp, schemas.TranscriptionStreamRequest, providerName, request.Model)
		}

		// Large payload streaming passthrough — pipe raw upstream SSE to client
		if providerUtils.SetupStreamingPassthrough(ctx, resp) {
			responseChan := make(chan *schemas.BifrostStreamChunk)
			close(responseChan)
			return responseChan, nil
		}

		// Create response channel
		responseChan := make(chan *schemas.BifrostStreamChunk, schemas.DefaultStreamBufferSize)

		providerUtils.SetStreamIdleTimeoutIfEmpty(ctx, provider.networkConfig.StreamIdleTimeoutInSeconds)

		// Start streaming in a goroutine
		go func() {
			defer providerUtils.EnsureStreamFinalizerCalled(ctx)
			defer func() {
				if ctx.Err() == context.Canceled {
					providerUtils.HandleStreamCancellation(ctx, postHookRunner, responseChan, providerName, request.Model, schemas.TranscriptionStreamRequest, logger)
				} else if ctx.Err() == context.DeadlineExceeded {
					providerUtils.HandleStreamTimeout(ctx, postHookRunner, responseChan, providerName, request.Model, schemas.TranscriptionStreamRequest, logger)
				}
				close(responseChan)
			}()
			defer providerUtils.ReleaseStreamingResponse(resp)
			// Decompress gzip-encoded streams transparently (no-op for non-gzip)
			reader, releaseGzip := providerUtils.DecompressStreamBody(resp)
			defer releaseGzip()

			// Wrap reader with idle timeout to detect stalled streams.
			reader, stopIdleTimeout := providerUtils.NewIdleTimeoutReader(reader, resp.BodyStream(), providerUtils.GetStreamIdleTimeout(ctx))
			defer stopIdleTimeout()

			// Setup cancellation handler to close the raw network stream on ctx cancellation,
			// which immediately unblocks any in-progress read (including reads blocked inside a gzip decompression layer).
			stopCancellation := providerUtils.SetupStreamCancellation(ctx, resp.BodyStream(), logger)
			defer stopCancellation()

			sseReader := providerUtils.GetSSEDataReader(ctx, reader)
			chunkIndex := -1

			startTime := time.Now()
			lastChunkTime := startTime
			var fullTranscriptionText strings.Builder

			for {
				// If context was cancelled/timed out, let defer handle it
				if ctx.Err() != nil {
					return
				}

				dataBytes, readErr := sseReader.ReadDataLine()
				if readErr != nil {
					if readErr != io.EOF {
						// If context was cancelled/timed out, let defer handle it
						if ctx.Err() != nil {
							return
						}
						ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
						logger.Warn("Error reading stream: %v", readErr)
						providerUtils.ProcessAndSendError(ctx, postHookRunner, readErr, responseChan, schemas.TranscriptionStreamRequest, providerName, request.Model, logger)
					}
					break
				}

				jsonData := string(dataBytes)

				// Skip empty data
				if strings.TrimSpace(jsonData) == "" {
					continue
				}

				var response schemas.BifrostTranscriptionStreamResponse
				var bifrostErr *schemas.BifrostError

				_, _, bifrostErr = HandleEWResponse(dataBytes, &response, nil, false, false)
				if bifrostErr != nil {
					bifrostErr.ExtraFields = schemas.BifrostErrorExtraFields{
						Provider:       providerName,
						ModelRequested: request.Model,
						RequestType:    schemas.TranscriptionStreamRequest,
					}
					ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
					providerUtils.ProcessAndSendBifrostError(ctx, postHookRunner, providerUtils.EnrichError(ctx, bifrostErr, body.Bytes(), dataBytes, false, providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse)), responseChan, logger)
					return
				}

				customChunk, ok := parseEWTranscriptionStreamChunk(dataBytes)
				if !ok || customChunk == nil {
					logger.Warn("customChunkParser returned no chunk")
					continue
				}
				response = *customChunk

				chunkIndex++
				if response.Delta != nil {
					fullTranscriptionText.WriteString(*response.Delta)
				}

				response.ExtraFields = schemas.BifrostResponseExtraFields{
					RequestType:    schemas.TranscriptionStreamRequest,
					Provider:       providerName,
					ModelRequested: request.Model,
					ChunkIndex:     chunkIndex,
					Latency:        time.Since(lastChunkTime).Milliseconds(),
				}
				lastChunkTime = time.Now()

				if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
					response.ExtraFields.RawResponse = jsonData
				}
				if response.Usage != nil || response.Type == schemas.TranscriptionStreamResponseTypeDone {
					response.ExtraFields.Latency = time.Since(startTime).Milliseconds()
					response.Text = fullTranscriptionText.String()
					ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
					providerUtils.ProcessAndSendResponse(ctx, postHookRunner, providerUtils.GetBifrostResponseForStreamResponse(nil, nil, nil, nil, &response, nil), responseChan)
					return
				}

				providerUtils.ProcessAndSendResponse(ctx, postHookRunner, providerUtils.GetBifrostResponseForStreamResponse(nil, nil, nil, nil, &response, nil), responseChan)
			}
		}()

		return responseChan, nil
	}
}

// ImageGeneration performs an image generation request to EW's API.
func (provider *EWProvider) ImageGeneration(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostImageGenerationRequest) (*schemas.BifrostImageGenerationResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	return openai.HandleOpenAIImageGenerationRequest(
		ctx,
		provider.client,
		baseURL+"/v1/images/generations",
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		provider.GetProviderKey(),
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		provider.logger,
	)
}

// ImageGenerationStream is not supported by the EW provider.
func (provider *EWProvider) ImageGenerationStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostImageGenerationRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ImageGenerationStreamRequest, provider.GetProviderKey())
}

// ImageEdit performs an image edit request to EW's API.
func (provider *EWProvider) ImageEdit(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostImageEditRequest) (*schemas.BifrostImageGenerationResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	return openai.HandleOpenAIImageEditRequest(
		ctx,
		provider.client,
		baseURL+"/v1/images/edits",
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		provider.GetProviderKey(),
		provider.logger,
	)
}

// ImageEditStream is not supported by the EW provider.
func (provider *EWProvider) ImageEditStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostImageEditRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ImageEditStreamRequest, provider.GetProviderKey())
}

// ImageVariation performs an image variation request to EW's API.
func (provider *EWProvider) ImageVariation(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostImageVariationRequest) (*schemas.BifrostImageGenerationResponse, *schemas.BifrostError) {
	baseURL, bifrostErr := provider.baseURLOrError(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	return openai.HandleOpenAIImageVariationRequest(
		ctx,
		provider.client,
		baseURL+"/v1/images/variations",
		request,
		key,
		provider.networkConfig.ExtraHeaders,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		provider.GetProviderKey(),
		provider.logger,
	)
}

// VideoGeneration is not supported by the EW provider.
func (provider *EWProvider) VideoGeneration(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostVideoGenerationRequest) (*schemas.BifrostVideoGenerationResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.VideoGenerationRequest, provider.GetProviderKey())
}

// VideoRetrieve is not supported by the EW provider.
func (provider *EWProvider) VideoRetrieve(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostVideoRetrieveRequest) (*schemas.BifrostVideoGenerationResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.VideoRetrieveRequest, provider.GetProviderKey())
}

// VideoDownload is not supported by the EW provider.
func (provider *EWProvider) VideoDownload(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostVideoDownloadRequest) (*schemas.BifrostVideoDownloadResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.VideoDownloadRequest, provider.GetProviderKey())
}

// VideoDelete is not supported by the EW provider.
func (provider *EWProvider) VideoDelete(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostVideoDeleteRequest) (*schemas.BifrostVideoDeleteResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.VideoDeleteRequest, provider.GetProviderKey())
}

// VideoList is not supported by the EW provider.
func (provider *EWProvider) VideoList(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostVideoListRequest) (*schemas.BifrostVideoListResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.VideoListRequest, provider.GetProviderKey())
}

// VideoRemix is not supported by the EW provider.
func (provider *EWProvider) VideoRemix(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostVideoRemixRequest) (*schemas.BifrostVideoGenerationResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.VideoRemixRequest, provider.GetProviderKey())
}

// FileUpload is not supported by the EW provider.
func (provider *EWProvider) FileUpload(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostFileUploadRequest) (*schemas.BifrostFileUploadResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.FileUploadRequest, provider.GetProviderKey())
}

// FileList is not supported by the EW provider.
func (provider *EWProvider) FileList(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostFileListRequest) (*schemas.BifrostFileListResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.FileListRequest, provider.GetProviderKey())
}

// FileRetrieve is not supported by the EW provider.
func (provider *EWProvider) FileRetrieve(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostFileRetrieveRequest) (*schemas.BifrostFileRetrieveResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.FileRetrieveRequest, provider.GetProviderKey())
}

// FileDelete is not supported by the EW provider.
func (provider *EWProvider) FileDelete(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostFileDeleteRequest) (*schemas.BifrostFileDeleteResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.FileDeleteRequest, provider.GetProviderKey())
}

// FileContent is not supported by the EW provider.
func (provider *EWProvider) FileContent(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostFileContentRequest) (*schemas.BifrostFileContentResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.FileContentRequest, provider.GetProviderKey())
}

// BatchCreate is not supported by the EW provider.
func (provider *EWProvider) BatchCreate(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostBatchCreateRequest) (*schemas.BifrostBatchCreateResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchCreateRequest, provider.GetProviderKey())
}

// BatchList is not supported by the EW provider.
func (provider *EWProvider) BatchList(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostBatchListRequest) (*schemas.BifrostBatchListResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchListRequest, provider.GetProviderKey())
}

// BatchRetrieve is not supported by the EW provider.
func (provider *EWProvider) BatchRetrieve(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostBatchRetrieveRequest) (*schemas.BifrostBatchRetrieveResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchRetrieveRequest, provider.GetProviderKey())
}

// BatchCancel is not supported by the EW provider.
func (provider *EWProvider) BatchCancel(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostBatchCancelRequest) (*schemas.BifrostBatchCancelResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchCancelRequest, provider.GetProviderKey())
}

// BatchDelete is not supported by the EW provider.
func (provider *EWProvider) BatchDelete(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostBatchDeleteRequest) (*schemas.BifrostBatchDeleteResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchDeleteRequest, provider.GetProviderKey())
}

// BatchResults is not supported by the EW provider.
func (provider *EWProvider) BatchResults(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostBatchResultsRequest) (*schemas.BifrostBatchResultsResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchResultsRequest, provider.GetProviderKey())
}

// CountTokens is not supported by the EW provider.
func (provider *EWProvider) CountTokens(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostResponsesRequest) (*schemas.BifrostCountTokensResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.CountTokensRequest, provider.GetProviderKey())
}

// ContainerCreate is not supported by the EW provider.
func (provider *EWProvider) ContainerCreate(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostContainerCreateRequest) (*schemas.BifrostContainerCreateResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerCreateRequest, provider.GetProviderKey())
}

// ContainerList is not supported by the EW provider.
func (provider *EWProvider) ContainerList(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerListRequest) (*schemas.BifrostContainerListResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerListRequest, provider.GetProviderKey())
}

// ContainerRetrieve is not supported by the EW provider.
func (provider *EWProvider) ContainerRetrieve(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerRetrieveRequest) (*schemas.BifrostContainerRetrieveResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerRetrieveRequest, provider.GetProviderKey())
}

// ContainerDelete is not supported by the EW provider.
func (provider *EWProvider) ContainerDelete(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerDeleteRequest) (*schemas.BifrostContainerDeleteResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerDeleteRequest, provider.GetProviderKey())
}

// ContainerFileCreate is not supported by the EW provider.
func (provider *EWProvider) ContainerFileCreate(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostContainerFileCreateRequest) (*schemas.BifrostContainerFileCreateResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerFileCreateRequest, provider.GetProviderKey())
}

// ContainerFileList is not supported by the EW provider.
func (provider *EWProvider) ContainerFileList(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerFileListRequest) (*schemas.BifrostContainerFileListResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerFileListRequest, provider.GetProviderKey())
}

// ContainerFileRetrieve is not supported by the EW provider.
func (provider *EWProvider) ContainerFileRetrieve(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerFileRetrieveRequest) (*schemas.BifrostContainerFileRetrieveResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerFileRetrieveRequest, provider.GetProviderKey())
}

// ContainerFileContent is not supported by the EW provider.
func (provider *EWProvider) ContainerFileContent(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerFileContentRequest) (*schemas.BifrostContainerFileContentResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerFileContentRequest, provider.GetProviderKey())
}

// ContainerFileDelete is not supported by the EW provider.
func (provider *EWProvider) ContainerFileDelete(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerFileDeleteRequest) (*schemas.BifrostContainerFileDeleteResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerFileDeleteRequest, provider.GetProviderKey())
}

// Passthrough is not supported by the EW provider.
func (provider *EWProvider) Passthrough(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostPassthroughRequest) (*schemas.BifrostPassthroughResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.PassthroughRequest, provider.GetProviderKey())
}

func (provider *EWProvider) PassthroughStream(_ *schemas.BifrostContext, _ schemas.PostHookRunner, _ schemas.Key, _ *schemas.BifrostPassthroughRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.PassthroughStreamRequest, provider.GetProviderKey())
}