package ew

// EWRerankRequest is the EW rerank request body.
type EWRerankRequest struct {
	Model           string                 `json:"model"`
	Query           string                 `json:"query"`
	Documents       []string               `json:"documents"`
	TopN            *int                   `json:"top_n,omitempty"`
	MaxTokensPerDoc *int                   `json:"max_tokens_per_doc,omitempty"`
	Priority        *int                   `json:"priority,omitempty"`
	ExtraParams     map[string]interface{} `json:"-"`
}

// GetExtraParams returns passthrough parameters for providerUtils.CheckContextAndGetRequestBody.
func (r *EWRerankRequest) GetExtraParams() map[string]interface{} {
	return r.ExtraParams
}