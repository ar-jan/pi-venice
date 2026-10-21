/**
 * Venice AI provider for Pi.
 *
 * - Auth: `/login venice` (stored in auth.json) or the VENICE_API_KEY environment variable.
 * - Models: fetched from Venice's live catalog (`GET /models?type=text`), limited to models that
 *   support tool calling, and cached in models-store.json for offline use.
 * - Every model name is marked "(private)" or "(anonymized)" after Venice's privacy tier.
 * - Requests disable Venice's own system prompt and carry a per-session `prompt_cache_key`
 *   (see request.ts).
 */

import {
	type Api,
	clampThinkingLevel,
	createProvider,
	envApiKeyAuth,
	type Model,
	type ThinkingLevelMap,
} from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { withVeniceRequestDefaults } from "./request.ts";

const PROVIDER_ID = "venice";
const BASE_URL = "https://api.venice.ai/api/v1";
const MODELS_TIMEOUT_MS = 15_000;

// =============================================================================
// Venice catalog
// =============================================================================

interface VenicePrice {
	usd?: number;
}

interface VeniceModel {
	id: string;
	type?: string;
	model_spec?: {
		name?: string;
		privacy?: string;
		offline?: boolean;
		availableContextTokens?: number;
		maxCompletionTokens?: number;
		deprecation?: { date?: string; removesAt?: string } | null;
		capabilities?: {
			supportsFunctionCalling?: boolean;
			supportsReasoning?: boolean;
			supportsReasoningEffort?: boolean;
			reasoningEffortOptions?: string[];
			supportsVision?: boolean;
			supportsMultipleImages?: boolean;
			supportsE2EE?: boolean;
			supportsTeeAttestation?: boolean;
			maxImages?: number;
		};
		pricing?: {
			input?: VenicePrice;
			output?: VenicePrice;
			cache_input?: VenicePrice;
			cache_write?: VenicePrice;
			extended?: {
				context_token_threshold?: number;
				input?: VenicePrice;
				output?: VenicePrice;
				cache_input?: VenicePrice;
				cache_write?: VenicePrice;
			};
		};
	};
}

const EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"] as const;

function thinkingConfig(model: VeniceModel): Pick<Model<Api>, "reasoning" | "thinkingLevelMap"> & {
	supportsReasoningEffort: boolean;
} {
	const caps = model.model_spec?.capabilities ?? {};
	if (!caps.supportsReasoning) {
		return { reasoning: false, supportsReasoningEffort: false };
	}
	const options = caps.supportsReasoningEffort ? (caps.reasoningEffortOptions ?? []) : [];
	if (options.length === 0) {
		// Reasons, but Venice does not accept reasoning_effort: expose one level meaning "model default".
		return {
			reasoning: true,
			supportsReasoningEffort: false,
			thinkingLevelMap: { off: null, minimal: null, low: null, high: null },
		};
	}
	const map: ThinkingLevelMap = {
		off: options.includes("none") ? "none" : null,
		minimal: options.includes("minimal") ? "minimal" : null,
	};
	for (const level of EFFORT_LEVELS) {
		map[level] = options.includes(level) ? level : null;
	}
	return { reasoning: true, supportsReasoningEffort: true, thinkingLevelMap: map };
}

function usd(price: VenicePrice | undefined): number {
	return typeof price?.usd === "number" ? price.usd : 0;
}

function toPiModel(model: VeniceModel): Model<"openai-completions"> | undefined {
	const spec = model.model_spec;
	const caps = spec?.capabilities;
	if (!spec || !caps) return undefined;
	if (model.type !== undefined && model.type !== "text") return undefined;
	if (!caps.supportsFunctionCalling || spec.offline) return undefined;

	const now = Date.now();
	const removesAt = spec.deprecation?.removesAt ?? spec.deprecation?.date;
	if (removesAt && Date.parse(removesAt) <= now) return undefined;

	const privacy = spec.privacy === "private" ? "private" : "anonymized";
	const tags: string[] = [privacy];
	if (caps.supportsE2EE || caps.supportsTeeAttestation) tags.push("TEE");
	if (removesAt) tags.push(`deprecated ${removesAt.slice(0, 10)}`);

	const pricing = spec.pricing ?? {};
	const extended = pricing.extended;
	const thinking = thinkingConfig(model);
	const vision = caps.supportsVision === true;
	const maxImages = caps.supportsMultipleImages === false ? 1 : caps.maxImages;

	return {
		id: model.id,
		name: `${spec.name ?? model.id} (${tags.join(", ")})`,
		api: "openai-completions",
		provider: PROVIDER_ID,
		baseUrl: BASE_URL,
		input: vision ? ["text", "image"] : ["text"],
		...(vision && maxImages ? { inputLimits: { images: { maxPerMessage: maxImages } } } : {}),
		cost: {
			input: usd(pricing.input),
			output: usd(pricing.output),
			cacheRead: usd(pricing.cache_input),
			cacheWrite: usd(pricing.cache_write),
			...(extended?.context_token_threshold
				? {
						tiers: [
							{
								inputTokensAbove: extended.context_token_threshold,
								input: usd(extended.input ?? pricing.input),
								output: usd(extended.output ?? pricing.output),
								cacheRead: usd(extended.cache_input ?? pricing.cache_input),
								cacheWrite: usd(extended.cache_write ?? pricing.cache_write),
							},
						],
					}
				: {}),
		},
		reasoning: thinking.reasoning,
		...(thinking.thinkingLevelMap ? { thinkingLevelMap: thinking.thinkingLevelMap } : {}),
		...(pricing.cache_input ? { promptCache: { short: 300 } } : {}),
		contextWindow: spec.availableContextTokens ?? 128_000,
		maxTokens: spec.maxCompletionTokens ?? 16_384,
		compat: {
			thinkingFormat: "openai",
			supportsReasoningEffort: thinking.supportsReasoningEffort,
			maxTokensField: "max_completion_tokens",
			supportsStore: false,
			supportsDeveloperRole: false,
			supportsUsageInStreaming: true,
			supportsLongCacheRetention: false,
			supportsStrictMode: false,
		},
	};
}

async function fetchVeniceModels(apiKey: string | undefined, signal: AbortSignal): Promise<Model<"openai-completions">[]> {
	const headers: Record<string, string> = { accept: "application/json" };
	if (apiKey) headers.authorization = `Bearer ${apiKey}`;
	const response = await fetch(`${BASE_URL}/models?type=text`, {
		headers,
		signal: AbortSignal.any([signal, AbortSignal.timeout(MODELS_TIMEOUT_MS)]),
	});
	if (!response.ok) throw new Error(`Venice model list request failed: ${response.status}`);
	const body = (await response.json()) as { data?: unknown };
	if (!Array.isArray(body.data)) throw new Error("Invalid Venice model list response");
	return (body.data as VeniceModel[])
		.map(toPiModel)
		.filter((model): model is Model<"openai-completions"> => model !== undefined)
		.sort((a, b) => a.name.localeCompare(b.name));
}

// =============================================================================
// Extension
// =============================================================================

export default function (pi: ExtensionAPI) {
	pi.registerProvider(
		createProvider({
			id: PROVIDER_ID,
			name: "Venice AI",
			baseUrl: BASE_URL,
			auth: { apiKey: envApiKeyAuth("Venice API key", ["VENICE_API_KEY"]) },
			models: [],
			fetchModels: (context) => {
				const key = context.credential?.type === "api_key" ? context.credential.key : undefined;
				return fetchVeniceModels(key, context.signal);
			},
			api: withVeniceRequestDefaults(openAICompletionsApi(), clampThinkingLevel),
		}),
	);
}
