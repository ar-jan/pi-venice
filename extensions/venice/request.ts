/**
 * Venice request defaults, applied to the final request body.
 *
 * Pi merges sampling parameters one level deep: the last of the model's `samplingParams`, the
 * thinking level's override, and the request's `samplingParams` that sets `venice_parameters`
 * replaces it whole. Merging `venice_parameters` here, key by key and after pi has assembled the
 * body, keeps the extension default and every user layer together.
 *
 * Thinking is turned off with `venice_parameters.disable_thinking` for the models whose entry
 * offers "off" without a wire value (see DISABLE_THINKING_MODELS in index.ts). Models that list a
 * `"none"` effort keep sending `reasoning_effort: "none"`, which Venice honors natively; on the
 * rest the toggle only strips the reasoning text while still billing it, so they do not offer "off".
 *
 * Venice reports cache writes as `prompt_tokens_details.cache_creation_input_tokens`, while pi's
 * OpenAI-completions adapter reads `prompt_tokens_details.cache_write_tokens`. The adapter therefore
 * counts written tokens as plain input and reports no cache-write cost. Wrapping the stream moves
 * them onto `cacheWrite` and recomputes the cost before pi sees each event.
 */

import { calculateCost, createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import type {
	Api,
	AssistantMessage,
	AssistantMessageEvent,
	AssistantMessageEventStream,
	Model,
	ModelThinkingLevel,
	ProviderEnv,
	ProviderStreams,
	SamplingParams,
	StreamOptions,
	Usage,
} from "@earendil-works/pi-ai";

/** Venice keeps the first 64 characters, like OpenAI. */
const PROMPT_CACHE_KEY_MAX_LENGTH = 64;

/** pi-ai's `clampThinkingLevel`, passed in so this module does not import the model catalog. */
export type ClampThinkingLevel = (model: Model<Api>, level: ModelThinkingLevel) => ModelThinkingLevel;

export interface VeniceRequestContext {
	sessionId?: string;
	cacheRetention?: StreamOptions["cacheRetention"];
	env?: ProviderEnv;
	/** User `venice_parameters` values in pi's merge order: model, thinking level, request. */
	veniceParameterLayers?: readonly unknown[];
	/** The effective thinking level is off, so ask Venice to disable and strip reasoning. */
	disableThinking?: boolean;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
	const proto = Object.getPrototypeOf(value);
	return proto === Object.prototype || proto === null;
}

// =============================================================================
// Cache-write usage
// =============================================================================

/** Raw cache-write count seen on the provider's chunks for the current request. */
interface CacheWriteObservation {
	writes: number;
}

function emptyUsage(): Usage {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

/**
 * Cache-write tokens on a raw provider chunk, or `undefined` when the adapter already maps them.
 * Venice sends `prompt_tokens_details.cache_creation_input_tokens` and mirrors it at the top level.
 */
function rawCacheWrites(chunk: unknown): number | undefined {
	if (!isPlainObject(chunk) || !isPlainObject(chunk.usage)) return undefined;
	const usage = chunk.usage;
	const details = isPlainObject(usage.prompt_tokens_details) ? usage.prompt_tokens_details : undefined;
	// A provider that sends `cache_write_tokens` needs no correction, and counting both would double it.
	if (details && typeof details.cache_write_tokens === "number") return undefined;
	const writes = details?.cache_creation_input_tokens ?? usage.cache_creation_input_tokens;
	return typeof writes === "number" && writes > 0 ? writes : undefined;
}

function observeCacheWrites(chunk: unknown, observation: CacheWriteObservation): void {
	const writes = rawCacheWrites(chunk);
	if (writes !== undefined && writes > observation.writes) observation.writes = writes;
}

/**
 * Move the observed write tokens off `input` and onto `cacheWrite`, then recompute the cost with the
 * model's rates (Venice bills writes above plain input on Claude). Idempotent: usage that already
 * reports writes is left alone, so forwarding an object on several events patches it once.
 */
export function applyVeniceCacheWriteUsage(usage: Usage, model: Model<Api>, writes: number): void {
	if (writes <= 0 || usage.cacheWrite > 0) return;
	const moved = Math.min(writes, usage.input);
	if (moved <= 0) return;
	usage.input -= moved;
	usage.cacheWrite = moved;
	usage.totalTokens = usage.input + usage.output + usage.cacheRead + usage.cacheWrite;
	calculateCost(model, usage);
}

function eventUsage(event: AssistantMessageEvent): Usage {
	if (event.type === "done") return event.message.usage;
	if (event.type === "error") return event.error.usage;
	return event.partial.usage;
}

function eventMessage(event: AssistantMessageEvent): AssistantMessage {
	if (event.type === "done") return event.message;
	if (event.type === "error") return event.error;
	return event.partial;
}

/**
 * Forward a provider stream, correcting cache-write usage just before each event reaches pi. The
 * events carry the adapter's own message objects, so the same correction also reaches the final
 * `done`/`error` message and pi's cost accounting.
 */
export function withVeniceCacheWriteUsage(
	stream: AssistantMessageEventStream,
	model: Model<Api>,
	observation: CacheWriteObservation,
): AssistantMessageEventStream {
	const forwarded = createAssistantMessageEventStream();
	let last: AssistantMessage | undefined;
	void (async () => {
		try {
			for await (const event of stream) {
				last = eventMessage(event);
				applyVeniceCacheWriteUsage(eventUsage(event), model, observation.writes);
				forwarded.push(event);
			}
		} catch (error) {
			// pi waits for a terminal event to resolve the response. Adapters emit one themselves, so
			// this only covers a stream that threw without ending.
			const failed: AssistantMessage = {
				...(last ?? {
					role: "assistant",
					content: [],
					api: model.api,
					provider: model.provider,
					model: model.id,
					usage: emptyUsage(),
					timestamp: Date.now(),
				}),
				stopReason: "error",
				errorMessage: error instanceof Error ? error.message : String(error),
			};
			forwarded.push({ type: "error", reason: "error", error: failed });
		} finally {
			forwarded.end();
		}
	})();
	return forwarded;
}

/** Same resolution as pi's adapters: explicit option, then PI_CACHE_RETENTION=long, else short. */
function resolveCacheRetention(context: VeniceRequestContext): "none" | "short" | "long" {
	if (context.cacheRetention) return context.cacheRetention;
	const fromEnv = context.env?.PI_CACHE_RETENTION || process.env.PI_CACHE_RETENTION;
	return fromEnv === "long" ? "long" : "short";
}

function promptCacheKey(context: VeniceRequestContext): string | undefined {
	if (!context.sessionId || resolveCacheRetention(context) === "none") return undefined;
	return Array.from(context.sessionId).slice(0, PROMPT_CACHE_KEY_MAX_LENGTH).join("");
}

/**
 * Add Venice defaults to a chat completions body:
 * - `venice_parameters`: `include_venice_system_prompt: false` and, at thinking level off,
 *   `disable_thinking: true`, then each user layer key by key, then the body's own value, so later
 *   layers win per key. A non-object value in the body is left for Venice to reject; non-object
 *   layers are skipped.
 * - `prompt_cache_key` from the session ID, so Venice routes the session to servers holding its
 *   cached prefix. Skipped when the body already has one or prompt caching is off.
 */
export function applyVeniceDefaults(payload: unknown, context: VeniceRequestContext): unknown {
	if (!isPlainObject(payload)) return payload;
	let body = payload;

	const veniceParameters = body.venice_parameters;
	if (veniceParameters === undefined || isPlainObject(veniceParameters)) {
		const merged: Record<string, unknown> = { include_venice_system_prompt: false };
		if (context.disableThinking) merged.disable_thinking = true;
		for (const layer of [...(context.veniceParameterLayers ?? []), veniceParameters]) {
			if (isPlainObject(layer)) Object.assign(merged, layer);
		}
		body = { ...body, venice_parameters: merged };
	}

	const cacheKey = body.prompt_cache_key === undefined ? promptCacheKey(context) : undefined;
	if (cacheKey) body = { ...body, prompt_cache_key: cacheKey };

	return body;
}

/** The `venice_parameters` of each layer pi merges, for an already clamped thinking level. */
function veniceParameterLayers(
	model: Model<Api>,
	level: ModelThinkingLevel,
	requestParams: SamplingParams | undefined,
): unknown[] {
	return [
		model.samplingParams?.venice_parameters,
		model.samplingParamsByThinkingLevel?.[level]?.venice_parameters,
		requestParams?.venice_parameters,
	];
}

function withDefaults<T extends StreamOptions>(
	options: T | undefined,
	layers: readonly unknown[],
	disableThinking: boolean,
	observation: CacheWriteObservation,
): T {
	const callerOnPayload = options?.onPayload;
	const callerOnStreamEvent = options?.onProviderStreamEvent;
	return {
		...options,
		onPayload: async (payload, model) => {
			const prepared = applyVeniceDefaults(payload, {
				...options,
				veniceParameterLayers: layers,
				disableThinking,
			});
			// Extension `before_provider_request` hooks arrive here and see the prepared body.
			const replaced = await callerOnPayload?.(prepared, model);
			return replaced === undefined ? prepared : replaced;
		},
		onProviderStreamEvent: async (event, model) => {
			// Observation only: pi documents this hook's payload as read-only.
			observeCacheWrites(event, observation);
			await callerOnStreamEvent?.(event, model);
		},
	} as T;
}

/** Whether this request asks Venice's own toggle to turn thinking off. */
function thinkingOff(model: Model<Api>, level: ModelThinkingLevel): boolean {
	// "off" maps to a string ("none") where Venice accepts that effort, and pi sends it itself.
	// Only a level offered without a wire value needs the Venice parameter.
	return model.reasoning === true && level === "off" && model.thinkingLevelMap?.off === undefined;
}

/** Wrap an API implementation so every request through it gets the Venice defaults. */
export function withVeniceRequestDefaults(api: ProviderStreams, clamp: ClampThinkingLevel): ProviderStreams {
	return {
		...api,
		stream: (model, context, options) => {
			// The completions adapter reads `reasoningEffort`; pi uses "off" when it is unset.
			const requested = (options as { reasoningEffort?: ModelThinkingLevel } | undefined)?.reasoningEffort ?? "off";
			const level = clamp(model, requested);
			const layers = veniceParameterLayers(model, level, options?.samplingParams);
			const observation: CacheWriteObservation = { writes: 0 };
			return withVeniceCacheWriteUsage(
				api.stream(model, context, withDefaults(options, layers, thinkingOff(model, level), observation)),
				model,
				observation,
			);
		},
		streamSimple: (model, context, options) => {
			const level = clamp(model, options?.reasoning ?? "off");
			const layers = veniceParameterLayers(model, level, options?.samplingParams);
			const observation: CacheWriteObservation = { writes: 0 };
			return withVeniceCacheWriteUsage(
				api.streamSimple(model, context, withDefaults(options, layers, thinkingOff(model, level), observation)),
				model,
				observation,
			);
		},
	};
}
