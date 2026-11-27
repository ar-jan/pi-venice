/**
 * Venice AI provider for Pi.
 *
 * - Auth: `/login venice` (saved to Pi's active credential store) or the VENICE_API_KEY environment variable.
 *   Fallback keys use numbered variables (VENICE_API_KEY_2, ...); see keyring.ts.
 * - Models: fetched from Venice's live catalog (`GET /models?type=text`), limited to models that
 *   support tool calling, and cached in models-store.json for offline use. E2EE-capable models are
 *   included and run in TEE-only mode: requests reach Venice as plaintext (see toPiModel).
 * - Decision models (`GET /models?type=decision`, e.g. `jev-latest`, `clef-api`) are registered as
 *   Pi `classifier` models using Pi's built-in TypeSafe System One client (`POST /systemone`),
 *   available through codemode `models.classify` and `ctx.modelRegistry.classify`. They do not appear
 *   in `/model`, and Venice accepts PNG/JPEG images only (no WebP) for them.
 * - Every model name is marked "(private)" or "(anonymized)" after Venice's privacy tier, plus
 *   "TEE" for models that run in an attested enclave.
 * - Requests disable Venice's own system prompt, carry a per-session `prompt_cache_key`, turn
 *   thinking off with `venice_parameters.disable_thinking` on the models verified to honor it, and
 *   place Claude cache breakpoints on the system prompt and the last turn (see request.ts).
 * - Usage: cache writes are moved onto `cacheWrite`, and the total cost is what Venice billed
 *   (see request.ts).
 * - `/venice-refresh-models` refetches the catalog on demand and applies it to the current model
 *   (see refresh.ts).
 */

import {
	type Api,
	clampThinkingLevel,
	type ClassifierModel,
	createProvider,
	isModelType,
	type Model,
	type Provider,
	type ProviderClassifier,
	type ProviderStreams,
	type RefreshModelsContext,
	type SystemOneCompat,
	type ThinkingLevelMap,
} from "@earendil-works/pi-ai";
// Pi's extension loader maps only a fixed set of pi-ai specifiers onto the host
// package (root, /compat, /oauth, /providers/all). Installed extensions have no
// node_modules, so the per-API modules ("@earendil-works/pi-ai/api/*") cannot be
// resolved here and fail at load time. Keep the compat import until pi exposes a
// stable entry point for API implementations.
import { openAICompletionsApi } from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { borrowedSystemOne } from "./classify.ts";
import {
	candidateKeys,
	describeKeys,
	Keyring,
	keyFingerprint,
	loadPreferredKeyHash,
	savePreferredKeyHash,
	veniceApiKeyAuth,
	withClassifierKeyRotation,
	withKeyRotation,
} from "./keyring.ts";
import { registerProviderWithRefreshCommand } from "./refresh.ts";
import { VENICE_MAX_IMAGES_PER_REQUEST, withVeniceRequestDefaults } from "./request.ts";

const PROVIDER_ID = "venice";
const BASE_URL = "https://api.venice.ai/api/v1";
const MODELS_TIMEOUT_MS = 15_000;
/**
 * Shorter timeout for the optional beta decision catalog so a stalled `/models?type=decision`
 * request falls back to cached classifiers well before Pi's 15-second interactive refresh deadline.
 */
export const DECISION_MODELS_TIMEOUT_MS = 5_000;

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
		maxStateTokens?: number;
		maxTotalTokens?: number;
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

type VeniceCapabilities = NonNullable<NonNullable<VeniceModel["model_spec"]>["capabilities"]>;

const EFFORT_LEVELS = ["low", "medium", "high", "xhigh", "max"] as const;

const DEFAULT_CONTEXT_WINDOW = 128_000;
const DEFAULT_MAX_TOKENS = 16_384;
const DEFAULT_DECISION_CONTEXT_WINDOW = 32_768;

// Catalogs sometimes publish an output cap equal to (or above) the context window when the real cap
// is unknown. Taken at face value it asks for the whole window as output and leaves nothing for
// input. Venice's live catalog is well formed (no model reaches even half its window), so this only
// fires on bad data; keep at least three quarters of the window for input.
const MAX_PLAUSIBLE_OUTPUT_TOKENS = 32_000;

function modelLimits(spec: VeniceModel["model_spec"]): { contextWindow: number; maxTokens: number } {
	const contextWindow = spec?.availableContextTokens ?? DEFAULT_CONTEXT_WINDOW;
	const maxTokens = spec?.maxCompletionTokens ?? DEFAULT_MAX_TOKENS;
	if (contextWindow <= 0 || maxTokens < contextWindow) return { contextWindow, maxTokens };
	return {
		contextWindow,
		maxTokens: Math.min(MAX_PLAUSIBLE_OUTPUT_TOKENS, Math.max(1, Math.floor(contextWindow / 4))),
	};
}

// Models that list no "none" effort but were verified against the live API to actually stop
// reasoning when Venice is asked to disable thinking: with `venice_parameters.disable_thinking`
// their completion tokens drop to the answer alone (or Venice reports reasoning_tokens 0).
// On every other such model Venice only strips the reasoning text while still generating and
// billing it, so offering "off" there would promise something the request does not deliver.
// Models that instead write their reasoning into the visible answer once thinking is off
// (claude-opus-4-5, claude-sonnet-4-6) are left out too: "off" would not shorten the reply.
// Re-verify when Venice changes these models or moves them to another backend.
const DISABLE_THINKING_MODELS = new Set([
	"abliteration-abliterated-model-large-flash",
	"claude-opus-4-6",
	"claude-opus-4-7",
	"claude-opus-4-8",
	"gemini-3-5-flash-lite",
	"grok-4-20",
	"minimax-m3-1-flash",
	"minimax-m3-preview",
	"nvidia-nemotron-3-5-super-vl-120b-a12b",
	"qwen-3-6-plus",
	"qwen-3-7-max",
	"qwen-3-7-plus",
	"qwen-3-8-flash",
	"qwen3-6-35b-a3b",
	"qwen3-235b-a22b-thinking-2507",
	"seed-2-1-turbo",
	"zai-org-glm-4.6",
	"zai-org-glm-4.7",
]);

// Models with effort control that do not list "none" but were verified against the live API to
// stop reasoning when sent `reasoning_effort: "none"` anyway, while `disable_thinking` only hides
// their reasoning (kimi-k2-7-code keeps billing about 190 reasoning tokens with it). "off" maps to
// "none" there, which pi sends itself. Left out: minimax-m25, which with "none" sometimes reasons in
// the visible answer until the token limit. Re-verify when Venice changes these models.
const NONE_EFFORT_MODELS = new Set(["kimi-k2-7-code"]);

// Models the catalog lists with tool calling whose backend rejects every request that carries tools,
// verified against the live API. e2ee-qwen3-vl-30b-a3b-p answers HTTP 400 "\"auto\" tool choice
// requires --enable-auto-tool-choice and --tool-call-parser to be set" (and the equivalent for
// "required" and a named tool); only tool_choice "none" works. Pi sends its tools with every agent
// request, so such a model would fail every turn. Re-verify when Venice changes these models.
const BROKEN_TOOL_CALLING_MODELS = new Set(["e2ee-qwen3-vl-30b-a3b-p"]);

/**
 * Whether a listed "none" effort really turns reasoning off. Not on E2EE-capable models, which this
 * provider runs in TEE-only mode: verified against the live API on every such model that lists
 * "none" (e2ee-glm-5-2-p, e2ee-glm-5-3-p, e2ee-kimi-k3-p), each way of disabling reasoning
 * (`reasoning_effort: "none"`, `reasoning.enabled: false`, `venice_parameters.disable_thinking`)
 * fails with HTTP 400 "reasoning.max_tokens must be positive", while every listed effort level works.
 * Offering "off" there would fail each request, so it stays unavailable. Re-verify when Venice
 * changes these models.
 */
function honorsNoneEffort(caps: VeniceCapabilities): boolean {
	return caps.supportsE2EE !== true;
}

function thinkingConfig(model: VeniceModel): Pick<Model<Api>, "reasoning" | "thinkingLevelMap"> & {
	supportsReasoningEffort: boolean;
} {
	const caps = model.model_spec?.capabilities ?? {};
	if (!caps.supportsReasoning) {
		return { reasoning: false, supportsReasoningEffort: false };
	}
	const options = caps.supportsReasoningEffort ? (caps.reasoningEffortOptions ?? []) : [];
	// "none" is an effort Venice accepts itself (except in TEE-only mode, see honorsNoneEffort), and
	// on NONE_EFFORT_MODELS even unlisted; pi only sends an effort where the model has effort control.
	// Otherwise "off" is offered only for verified models, and without a map value: pi then sends no
	// reasoning_effort, and the request layer adds venice_parameters.disable_thinking (see
	// request.ts). null keeps it unsupported everywhere else.
	const acceptsNone =
		(options.includes("none") && honorsNoneEffort(caps)) ||
		(options.length > 0 && NONE_EFFORT_MODELS.has(model.id));
	const off = acceptsNone ? "none" : DISABLE_THINKING_MODELS.has(model.id) ? undefined : null;
	if (options.length === 0) {
		// No effort control: "medium" stays unset, which pi reads as a single "model default" level.
		return {
			reasoning: true,
			supportsReasoningEffort: false,
			thinkingLevelMap: { off, minimal: null, low: null, high: null },
		};
	}
	const map: ThinkingLevelMap = {
		off,
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

function modelTags(spec: NonNullable<VeniceModel["model_spec"]>, removesAt: string | undefined): string[] {
	const privacy = spec.privacy === "private" ? "private" : "anonymized";
	const tags: string[] = [privacy];
	if (spec.capabilities?.supportsTeeAttestation) tags.push("TEE");
	if (removesAt) tags.push(`deprecated ${removesAt.slice(0, 10)}`);
	return tags;
}

function modelCost(pricing: NonNullable<VeniceModel["model_spec"]>["pricing"]): Model<"openai-completions">["cost"] {
	const rates = pricing ?? {};
	const extended = rates.extended;
	return {
		input: usd(rates.input),
		output: usd(rates.output),
		cacheRead: usd(rates.cache_input),
		cacheWrite: usd(rates.cache_write),
		...(extended?.context_token_threshold
			? {
					tiers: [
						{
							inputTokensAbove: extended.context_token_threshold,
							input: usd(extended.input ?? rates.input),
							output: usd(extended.output ?? rates.output),
							cacheRead: usd(extended.cache_input ?? rates.cache_input),
							cacheWrite: usd(extended.cache_write ?? rates.cache_write),
						},
					],
				}
			: {}),
	};
}

export function toPiModel(model: VeniceModel): Model<"openai-completions"> | undefined {
	const spec = model.model_spec;
	const caps = spec?.capabilities;
	if (!spec || !caps) return undefined;
	if (model.type !== undefined && model.type !== "text") return undefined;
	if (!caps.supportsFunctionCalling || spec.offline) return undefined;
	if (BROKEN_TOOL_CALLING_MODELS.has(model.id)) return undefined;
	// E2EE-capable models stay. Venice runs them in TEE-only mode unless a request carries its E2EE
	// headers (`X-Venice-TEE-*`) with client-encrypted messages, which this provider never sends: the
	// model runs in an attested enclave, but the request reaches Venice as plaintext like any other.
	// Their names are therefore tagged "TEE" below, never "E2EE". The extension does not verify the
	// enclave's attestation itself; the tag reflects Venice's catalog.

	const now = Date.now();
	const removesAt = spec.deprecation?.removesAt ?? spec.deprecation?.date;
	if (removesAt && Date.parse(removesAt) <= now) return undefined;

	const tags = modelTags(spec, removesAt);
	const pricing = spec.pricing ?? {};
	const thinking = thinkingConfig(model);
	const { contextWindow, maxTokens } = modelLimits(spec);
	const vision = caps.supportsVision === true;
	const maxImages = caps.supportsMultipleImages === false ? 1 : caps.maxImages;

	return {
		id: model.id,
		name: `${spec.name ?? model.id} (${tags.join(", ")})`,
		api: "openai-completions",
		provider: PROVIDER_ID,
		baseUrl: BASE_URL,
		input: vision ? ["text", "image"] : ["text"],
		...(vision
			? {
					inputLimits: {
						images: {
							...(maxImages ? { maxPerMessage: maxImages } : {}),
							maxPerRequest: VENICE_MAX_IMAGES_PER_REQUEST,
						},
					},
				}
			: {}),
		cost: modelCost(pricing),
		reasoning: thinking.reasoning,
		...(thinking.thinkingLevelMap ? { thinkingLevelMap: thinking.thinkingLevelMap } : {}),
		...(pricing.cache_input ? { promptCache: { short: 300 } } : {}),
		contextWindow,
		maxTokens,
		compat: {
			thinkingFormat: "openai",
			supportsReasoningEffort: thinking.supportsReasoningEffort,
			maxTokensField: "max_completion_tokens",
			supportsStore: false,
			// Venice documents the developer message and honors it across catalog backends (Claude,
			// OpenAI, Gemini, Qwen, DeepSeek, Kimi, Grok, ...).
			supportsDeveloperRole: true,
			supportsUsageInStreaming: true,
			supportsLongCacheRetention: false,
			supportsStrictMode: false,
		},
	};
}

/**
 * Map a Venice decision catalog entry (`GET /models?type=decision`) to a Pi classifier model.
 *
 * Venice exposes System One at `POST /api/v1/systemone` (alias of `/decisions`). `compat` is
 * always set explicitly so TypeSafe's defaults (`assets` + `TypeSafe-Danger-Zone-Images: true`)
 * never leak onto Venice: Jev uses `assets` and every other decision model (e.g. `clef-api`) uses
 * `images`, both without the opt-in header.
 */
export function toPiClassifierModel(model: VeniceModel): ClassifierModel<"typesafe-system-one"> | undefined {
	const spec = model.model_spec;
	if (!spec) return undefined;
	if (model.type !== undefined && model.type !== "decision") return undefined;
	if (spec.offline) return undefined;

	const now = Date.now();
	const removesAt = spec.deprecation?.removesAt ?? spec.deprecation?.date;
	if (removesAt && Date.parse(removesAt) <= now) return undefined;

	const tags = modelTags(spec, removesAt);
	const contextWindow =
		spec.maxTotalTokens && spec.maxTotalTokens > 0
			? spec.maxTotalTokens
			: spec.maxStateTokens && spec.maxStateTokens > 0
				? spec.maxStateTokens
				: spec.availableContextTokens && spec.availableContextTokens > 0
					? spec.availableContextTokens
					: DEFAULT_DECISION_CONTEXT_WINDOW;
	// Live decision entries omit `capabilities` while Venice's swagger documents image input for
	// both current models (`jev-latest` and `clef-api`); honor an explicit `supportsVision: false`.
	const vision = spec.capabilities?.supportsVision !== false;
	const compat: SystemOneCompat = {
		imageInput: model.id.startsWith("jev") ? "assets" : "images",
		sendImageOptInHeader: false,
	};

	return {
		type: "classifier",
		id: model.id,
		name: `${spec.name ?? model.id} (${tags.join(", ")})`,
		api: "typesafe-system-one",
		provider: PROVIDER_ID,
		baseUrl: BASE_URL,
		input: vision ? ["text", "image"] : ["text"],
		cost: modelCost(spec.pricing),
		contextWindow,
		compat,
	};
}

async function fetchModelList(
	url: string,
	headers: Record<string, string>,
	signal: AbortSignal,
	timeoutMs: number,
): Promise<VeniceModel[]> {
	const timeout = new AbortController();
	const timer = setTimeout(() => {
		timeout.abort(new DOMException(`The operation timed out after ${timeoutMs}ms`, "TimeoutError"));
	}, timeoutMs);
	timer.unref?.();
	try {
		const response = await fetch(url, {
			headers,
			signal: AbortSignal.any([signal, timeout.signal]),
		});
		if (!response.ok) throw new Error(`Venice model list request failed: ${response.status}`);
		const body = (await response.json()) as { data?: unknown };
		if (!Array.isArray(body.data)) throw new Error("Invalid Venice model list response");
		return body.data as VeniceModel[];
	} finally {
		clearTimeout(timer);
	}
}

function cachedClassifiers(stored: RefreshModelsContext["stored"]): ClassifierModel<"typesafe-system-one">[] {
	return (stored?.models ?? []).filter(
		(model): model is ClassifierModel<"typesafe-system-one"> =>
			model.provider === PROVIDER_ID && isModelType(model, "classifier"),
	);
}

async function fetchVeniceModels(
	{ baseUrl = BASE_URL, credential, signal, stored }: RefreshModelsContext,
	includeDecisions: boolean,
	decisionTimeoutMs = DECISION_MODELS_TIMEOUT_MS,
): Promise<(Model<"openai-completions"> | ClassifierModel<"typesafe-system-one">)[]> {
	const headers: Record<string, string> = { accept: "application/json" };
	if (credential?.type === "api_key" && credential.key) headers.authorization = `Bearer ${credential.key}`;
	const root = baseUrl.replace(/\/+$/, "");

	const textPromise = fetchModelList(`${root}/models?type=text`, headers, signal, MODELS_TIMEOUT_MS).then((entries) =>
		entries
			.map(toPiModel)
			.filter((model): model is Model<"openai-completions"> => model !== undefined)
			.sort((a, b) => a.name.localeCompare(b.name)),
	);
	if (!includeDecisions) return textPromise;

	// Decision models are beta: a failing or stalled decision-list request must not fail the text
	// catalog refresh or hold it past Pi's 15-second interactive deadline. Because `createProvider`
	// replaces the whole persisted catalog with `fetchModels`'s return value, keep previously
	// cached classifiers on non-abort failures rather than wiping them out of `models-store.json`.
	const decisionAbort = new AbortController();
	const decisionPromise = fetchModelList(
		`${root}/models?type=decision`,
		headers,
		AbortSignal.any([signal, decisionAbort.signal]),
		decisionTimeoutMs,
	)
		.then((entries) =>
			entries
				.map(toPiClassifierModel)
				.filter((model): model is ClassifierModel<"typesafe-system-one"> => model !== undefined)
				.sort((a, b) => a.name.localeCompare(b.name)),
		)
		.catch((error) => {
			if (signal.aborted) throw error;
			return cachedClassifiers(stored);
		});

	try {
		const [chatModels, decisionModels] = await Promise.all([textPromise, decisionPromise]);
		return [...chatModels, ...decisionModels];
	} finally {
		decisionAbort.abort();
	}
}

// =============================================================================
// Extension
// =============================================================================

/**
 * The request pipeline the provider registers: pi's OpenAI-completions adapter with the Venice
 * request defaults inside, and key rotation outside so every retried key gets the same defaults.
 */
export function veniceApi(keyring: Keyring): ProviderStreams {
	return withKeyRotation(withVeniceRequestDefaults(openAICompletionsApi(), clampThinkingLevel), keyring);
}

export interface VeniceProviderOptions {
	keyring?: Keyring;
	getRegistry?: () => ExtensionContext["modelRegistry"] | undefined;
	/** Injected System One classifier; defaults to `borrowedSystemOne()`. */
	systemOne?: ProviderClassifier | undefined;
	/** Timeout for the optional decision catalog request; defaults to `DECISION_MODELS_TIMEOUT_MS`. */
	decisionTimeoutMs?: number;
}

export function createVeniceProvider(options: VeniceProviderOptions = {}): Provider {
	const keyring = options.keyring ?? new Keyring();
	const getRegistry = options.getRegistry ?? (() => undefined);
	const systemOne = "systemOne" in options ? options.systemOne : borrowedSystemOne();
	const decisionTimeoutMs = options.decisionTimeoutMs ?? DECISION_MODELS_TIMEOUT_MS;

	const base = createProvider({
		id: PROVIDER_ID,
		name: "Venice AI",
		baseUrl: BASE_URL,
		auth: { apiKey: veniceApiKeyAuth(getRegistry) },
		models: [],
		fetchModels: (context) => fetchVeniceModels(context, systemOne !== undefined, decisionTimeoutMs),
		api: veniceApi(keyring),
		...(systemOne
			? { classifiers: { "typesafe-system-one": withClassifierKeyRotation(systemOne, keyring) } }
			: {}),
	});

	if (systemOne) return base;

	// Without a classifier implementation, strip cached classifiers before `createProvider` restores
	// `models-store.json` so `getModelsOfType`, `findOfType`, and `getAvailableOfType` all exclude
	// them while cached chat models stay usable.
	return {
		...base,
		refreshModels(context) {
			const stored = context.stored
				? { ...context.stored, models: context.stored.models.filter((model) => !isModelType(model, "classifier")) }
				: undefined;
			return base.refreshModels!({ ...context, stored });
		},
	};
}

export default function (pi: ExtensionAPI) {
	const keyring = new Keyring(loadPreferredKeyHash());

	// The registry belongs to the active session, which can change on reload or session replacement.
	let registry: ExtensionContext["modelRegistry"] | undefined;
	pi.on("session_start", (_event, ctx) => {
		registry = ctx.modelRegistry;
	});
	pi.on("session_shutdown", () => {
		registry = undefined;
	});

	registerProviderWithRefreshCommand(pi, createVeniceProvider({ keyring, getRegistry: () => registry }));

	pi.registerCommand("venice-key", {
		description: "Select the active Venice API key",
		handler: async (args, ctx) => {
			const auth = await ctx.modelRegistry.getProviderAuth(PROVIDER_ID).catch(() => undefined);
			const keys = candidateKeys(auth?.auth.apiKey, auth?.env);
			if (keys.length === 0) {
				ctx.ui.notify("No Venice API keys configured. Run /login venice or set VENICE_API_KEY.", "warning");
				return;
			}
			const current = keyring.select(keys);
			const choices = describeKeys(keys, auth?.env, auth?.source, current);
			const choose = (index: number) => {
				const choice = choices[index]!;
				keyring.choose(choice.key);
				savePreferredKeyHash(keyFingerprint(choice.key));
				ctx.ui.notify(`Active Venice API key: ${choice.label.replace(/^\S*\s*/, "")}`, "info");
			};

			const requested = args.trim();
			if (requested) {
				const index = Number(requested);
				if (!Number.isInteger(index) || index < 1 || index > keys.length) {
					ctx.ui.notify(`Choose a key number from 1 to ${keys.length}.`, "error");
					return;
				}
				choose(index - 1);
				return;
			}

			if (!ctx.hasUI) {
				ctx.ui.notify(choices.map((choice) => choice.label).join("\n"), "info");
				return;
			}
			const selected = await ctx.ui.select("Active Venice API key", choices.map((choice) => choice.label));
			if (!selected) return;
			const chosen = choices.findIndex((choice) => choice.label === selected);
			if (chosen >= 0) choose(chosen);
		},
	});
}
