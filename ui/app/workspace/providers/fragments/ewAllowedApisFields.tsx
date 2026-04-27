"use client";

import { FormControl, FormDescription, FormField, FormItem, FormLabel } from "@/components/ui/form";
import { Switch } from "@/components/ui/switch";
import { AllowedRequests } from "@/lib/types/config";
import { Control, useFormContext, useWatch } from "react-hook-form";

interface EWAllowedApisFieldsProps {
	control: Control<any>;
	// Path to the EWKeyConfig.allowed_requests field, e.g. "key.ew_key_config.allowed_requests".
	namePrefix: string;
}

// Subset of AllowedRequests that the EW provider actually implements in
// core/providers/ew/ew.go. Other operations always return UnsupportedOperationError,
// so toggling them would have no effect.
type EWAllowedKey = Extract<
	keyof AllowedRequests,
	| "list_models"
	| "chat_completion"
	| "chat_completion_stream"
	| "text_completion"
	| "text_completion_stream"
	| "responses"
	| "responses_stream"
	| "embedding"
	| "rerank"
	| "speech"
	| "speech_stream"
	| "transcription"
	| "transcription_stream"
	| "image_generation"
	| "image_edit"
	| "image_variation"
>;

const EW_REQUEST_TYPES: { key: EWAllowedKey; label: string }[] = [
	{ key: "list_models", label: "List Models" },
	{ key: "chat_completion", label: "Chat Completion" },
	{ key: "chat_completion_stream", label: "Chat Completion (Stream)" },
	{ key: "text_completion", label: "Text Completion" },
	{ key: "text_completion_stream", label: "Text Completion (Stream)" },
	{ key: "responses", label: "Responses" },
	{ key: "responses_stream", label: "Responses (Stream)" },
	{ key: "embedding", label: "Embedding" },
	{ key: "rerank", label: "Rerank" },
	{ key: "speech", label: "Speech" },
	{ key: "speech_stream", label: "Speech (Stream)" },
	{ key: "transcription", label: "Transcription" },
	{ key: "transcription_stream", label: "Transcription (Stream)" },
	{ key: "image_generation", label: "Image Generation" },
	{ key: "image_edit", label: "Image Edit" },
	{ key: "image_variation", label: "Image Variation" },
];

// Build a default AllowedRequests object with all EW-supported APIs enabled and the
// rest disabled. This is what we write to the form when the operator toggles the master
// switch from "all allowed" to "custom" — they then untick individual APIs.
function buildAllAllowed(): AllowedRequests {
	const base: AllowedRequests = {
		text_completion: false,
		text_completion_stream: false,
		chat_completion: false,
		chat_completion_stream: false,
		responses: false,
		responses_stream: false,
		embedding: false,
		speech: false,
		speech_stream: false,
		transcription: false,
		transcription_stream: false,
		image_generation: false,
		image_generation_stream: false,
		image_edit: false,
		image_edit_stream: false,
		image_variation: false,
		count_tokens: false,
		list_models: false,
		rerank: false,
		video_generation: false,
		video_retrieve: false,
		video_download: false,
		video_delete: false,
		video_list: false,
		video_remix: false,
		websocket_responses: false,
		realtime: false,
	};
	for (const { key } of EW_REQUEST_TYPES) {
		base[key] = true;
	}
	return base;
}

export function EWAllowedApisFields({ control, namePrefix }: EWAllowedApisFieldsProps) {
	const { setValue } = useFormContext();
	const allowed = useWatch({ control, name: namePrefix }) as AllowedRequests | null | undefined;
	const isUnrestricted = allowed === null || allowed === undefined;

	const onMasterToggle = (checked: boolean) => {
		// checked === true ⇒ "all APIs allowed" ⇒ store null (backend default = no restrictions).
		// checked === false ⇒ "custom" ⇒ initialize with all EW APIs enabled, operator unticks individually.
		setValue(namePrefix, checked ? null : buildAllAllowed(), { shouldDirty: true });
	};

	const leftColumn = EW_REQUEST_TYPES.slice(0, Math.ceil(EW_REQUEST_TYPES.length / 2));
	const rightColumn = EW_REQUEST_TYPES.slice(Math.ceil(EW_REQUEST_TYPES.length / 2));

	const renderToggle = ({ key, label }: { key: EWAllowedKey; label: string }) => (
		<FormField
			key={key}
			control={control}
			name={`${namePrefix}.${key}`}
			render={({ field }) => (
				<FormItem className="flex flex-row items-center justify-between rounded-lg border p-3">
					<FormLabel>{label}</FormLabel>
					<FormControl>
						<Switch checked={!!field.value} onCheckedChange={field.onChange} size="md" />
					</FormControl>
				</FormItem>
			)}
		/>
	);

	return (
		<div className="space-y-4">
			<div className="flex flex-row items-start justify-between gap-4 rounded-lg border p-3">
				<div className="space-y-1">
					<FormLabel>Allow All APIs</FormLabel>
					<FormDescription>
						When enabled, this key handles every API the EW provider supports. Disable to restrict the key — useful when a backend
						doesn&apos;t implement or returns errors for specific APIs (e.g. embedding, rerank).
					</FormDescription>
				</div>
				<Switch checked={isUnrestricted} onCheckedChange={onMasterToggle} size="md" />
			</div>

			{!isUnrestricted && (
				<div className="grid grid-cols-2 gap-4">
					<div className="space-y-3">{leftColumn.map(renderToggle)}</div>
					<div className="space-y-3">{rightColumn.map(renderToggle)}</div>
				</div>
			)}
		</div>
	);
}
