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
 * `"none"` effort, or are verified to honor it unlisted (NONE_EFFORT_MODELS), keep sending
 * `reasoning_effort: "none"`; on the rest the toggle only strips the reasoning text while still
 * billing it, so they do not offer "off".
 *
 * Venice reports cache writes as `prompt_tokens_details.cache_creation_input_tokens`, while pi's
 * OpenAI-completions adapter reads `prompt_tokens_details.cache_write_tokens`. The adapter therefore
 * counts written tokens as plain input and reports no cache-write cost. Wrapping the stream moves
 * them onto `cacheWrite` and recomputes the cost before pi sees each event.
 *
 * Venice also reports what it billed, as a top-level `cost` on the final usage chunk. That total is
 * what pi shows, because a catalog estimate from token counts misses charges outside the token
 * prices: web search adds about $0.01 a request, so a search turn can cost 20 times its tokens. The
 * per-category amounts stay catalog estimates (see applyVeniceReportedCost).
 *
 * Claude needs explicit cache breakpoints. Venice adds its own on a `system` message and on the
 * second-to-last user turn (tool results count as user turns), so the newest exchange is billed as
 * uncached input and then written to the cache on the next request: every tool result is paid
 * twice. Venice also adds no breakpoint for a `developer` message. Requests to Claude therefore get
 * breakpoints of their own on the instruction message and the last turn (see withCacheBreakpoints).
 *
 * Venice rejects a message string that contains a NUL byte (U+0000) with HTTP 400, "String must
 * not contain null bytes"; every other control character is accepted. A NUL in one message therefore
 * fails not only that turn but every later request in the session, because pi replays the whole
 * transcript each time. Tool results are the usual source, in either of two ways: a command that
 * reads a binary file (`cat` of an executable), or a program whose stdout carries raw bytes (e.g.
 * printing a decoded blob). Such bytes are dropped from `messages` before sending.
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

const NUL = "\u0000";

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
	/** The model needs explicit cache breakpoints (Claude); skipped when prompt caching is off. */
	cacheBreakpoints?: boolean;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
	const proto = Object.getPrototypeOf(value);
	return proto === Object.prototype || proto === null;
}

// =============================================================================
// Usage and billed cost
// =============================================================================

/** What the provider's raw chunks reported for the current request beyond what the adapter parses. */
interface UsageObservation {
	/** Cache-write tokens (Venice's `cache_creation_input_tokens`). */
	writes: number;
	/** Billed cost in dollars, once the final usage chunk has arrived. */
	billed?: number;
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

/**
 * The billed cost on a raw chunk, from Venice's top-level `cost`: a USD part (bundled credits
 * included as USD) and a DIEM part. Venice prices DIEM one to one with USD (every catalog price lists
 * the same `usd` and `diem` amount), so their sum is the request's cost in the dollars pi shows.
 * `undefined` when the chunk has no cost or any part is not a finite, non-negative number.
 */
function rawBilledCost(chunk: unknown): number | undefined {
	if (!isPlainObject(chunk) || !isPlainObject(chunk.cost)) return undefined;
	const parts = [chunk.cost.usd, chunk.cost.diem].filter((part) => part !== undefined);
	if (parts.length === 0) return undefined;
	let total = 0;
	for (const part of parts) {
		if (typeof part !== "number" || !Number.isFinite(part) || part < 0) return undefined;
		total += part;
	}
	return total;
}

function observeUsage(chunk: unknown, observation: UsageObservation): void {
	const writes = rawCacheWrites(chunk);
	if (writes !== undefined && writes > observation.writes) observation.writes = writes;
	const billed = rawBilledCost(chunk);
	if (billed !== undefined) observation.billed = billed;
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

/**
 * Make Venice's billed cost the usage total, as pi's types define it ("provider-reported billed cost
 * when supported, otherwise the catalog estimate") and as pi's adapter does for OpenRouter. The
 * per-category amounts keep their catalog estimates, so they can sum to less than the total when
 * Venice bills extras such as web search. Idempotent; without a reported cost the estimate stays.
 */
export function applyVeniceReportedCost(usage: Usage, billed: number | undefined): void {
	if (billed === undefined) return;
	usage.cost.total = billed;
	usage.cost.totalFromProvider = true;
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
 * Forward a provider stream, correcting usage just before each event reaches pi: cache-write tokens
 * first (which recomputes the estimate), then the billed total. The events carry the adapter's own
 * message objects, so the same correction also reaches the final `done`/`error` message and pi's
 * cost accounting.
 */
export function withVeniceUsage(
	stream: AssistantMessageEventStream,
	model: Model<Api>,
	observation: UsageObservation,
): AssistantMessageEventStream {
	const forwarded = createAssistantMessageEventStream();
	let last: AssistantMessage | undefined;
	void (async () => {
		try {
			for await (const event of stream) {
				last = eventMessage(event);
				const usage = eventUsage(event);
				applyVeniceCacheWriteUsage(usage, model, observation.writes);
				applyVeniceReportedCost(usage, observation.billed);
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

// =============================================================================
// Claude cache breakpoints
// =============================================================================

/** A fresh object per breakpoint, so a later payload hook cannot change them all at once. */
const ephemeralCache = () => ({ type: "ephemeral" });

/** Claude caches only up to explicit breakpoints; every other Venice backend caches on its own. */
function usesExplicitCacheBreakpoints(model: Model<Api>): boolean {
	return model.id.startsWith("claude-");
}

/**
 * A copy of `message` whose last text part carries a breakpoint, or `undefined` when it has no text
 * to mark. A string content becomes a single text part. An existing `cache_control` is kept.
 */
function withTextBreakpoint(message: Record<string, unknown>): Record<string, unknown> | undefined {
	const { content } = message;
	if (typeof content === "string") {
		return content.length > 0
			? { ...message, content: [{ type: "text", text: content, cache_control: ephemeralCache() }] }
			: undefined;
	}
	if (!Array.isArray(content)) return undefined;
	for (let i = content.length - 1; i >= 0; i--) {
		const part = content[i];
		if (!isPlainObject(part) || part.type !== "text") continue;
		if (part.cache_control !== undefined) return message;
		const parts = [...content];
		parts[i] = { ...part, cache_control: ephemeralCache() };
		return { ...message, content: parts };
	}
	return undefined;
}

/**
 * Place Claude cache breakpoints where Venice honors them (measured against the live API):
 * - The first `system`/`developer` message, on its last text part. Venice adds no breakpoint of its
 *   own for `developer`, so without this the first request of a session caches nothing.
 * - The last turn, so the whole prompt is written now and read back by the next request instead of
 *   being billed as uncached input first. Venice reads a breakpoint on a `tool` message only from the
 *   message itself and on a `user` message only from a text part, so each role gets its own form.
 *   Venice ignores breakpoints on images, so a turn ending in an image is cached up to its
 *   last text part. Messages of other roles and user messages without text are skipped, so the
 *   breakpoint lands on the latest turn that can carry one.
 *
 * Together with Venice's own breakpoint on the second-to-last user turn this stays within Claude's
 * limit of four. Messages are copied on write; the input is not mutated.
 */
function withCacheBreakpoints(body: Record<string, unknown>): Record<string, unknown> {
	if (!Array.isArray(body.messages)) return body;
	const messages: unknown[] = [...body.messages];

	const instruction = messages.findIndex(
		(message) => isPlainObject(message) && (message.role === "system" || message.role === "developer"),
	);
	if (instruction >= 0) {
		const marked = withTextBreakpoint(messages[instruction] as Record<string, unknown>);
		if (marked) messages[instruction] = marked;
	}

	for (let i = messages.length - 1; i > instruction; i--) {
		const message = messages[i];
		if (!isPlainObject(message)) continue;
		if (message.role === "tool") {
			if (message.cache_control === undefined) messages[i] = { ...message, cache_control: ephemeralCache() };
			break;
		}
		if (message.role === "user") {
			const marked = withTextBreakpoint(message);
			if (!marked) continue;
			messages[i] = marked;
			break;
		}
	}

	return { ...body, messages };
}

/**
 * Replace every string in `value` that holds a NUL byte, keeping untouched values by reference.
 * Only the array and plain-object shapes a chat body uses are walked; anything else passes through.
 */
function withoutNullBytes(value: unknown): unknown {
	if (typeof value === "string") return value.includes(NUL) ? value.replaceAll(NUL, "") : value;
	if (Array.isArray(value)) {
		let changed = false;
		const next = value.map((item) => {
			const clean = withoutNullBytes(item);
			changed ||= clean !== item;
			return clean;
		});
		return changed ? next : value;
	}
	if (isPlainObject(value)) {
		let changed = false;
		const next: Record<string, unknown> = {};
		for (const [key, item] of Object.entries(value)) {
			const clean = withoutNullBytes(item);
			changed ||= clean !== item;
			next[key] = clean;
		}
		return changed ? next : value;
	}
	return value;
}

/**
 * Add Venice defaults to a chat completions body:
 * - `messages`: NUL bytes dropped (see withoutNullBytes), so a tool result carrying binary bytes —
 *   read from a file or printed by a program — cannot make Venice reject the session's every later
 *   request.
 * - `venice_parameters`: `include_venice_system_prompt: false` and, at thinking level off,
 *   `disable_thinking: true`, then each user layer key by key, then the body's own value, so later
 *   layers win per key. A non-object value in the body is left for Venice to reject; non-object
 *   layers are skipped.
 * - `prompt_cache_key` from the session ID, so Venice routes the session to servers holding its
 *   cached prefix. Skipped when the body already has one or prompt caching is off.
 * - Claude cache breakpoints when `cacheBreakpoints` is set (see withCacheBreakpoints), unless
 *   prompt caching is off.
 */
export function applyVeniceDefaults(payload: unknown, context: VeniceRequestContext): unknown {
	if (!isPlainObject(payload)) return payload;
	let body = payload;

	const messages = body.messages;
	if (Array.isArray(messages)) {
		const clean = withoutNullBytes(messages);
		if (clean !== messages) body = { ...body, messages: clean };
	}

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

	if (context.cacheBreakpoints && resolveCacheRetention(context) !== "none") {
		body = withCacheBreakpoints(body);
	}

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
	observation: UsageObservation,
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
				cacheBreakpoints: usesExplicitCacheBreakpoints(model),
			});
			// Extension `before_provider_request` hooks arrive here and see the prepared body.
			const replaced = await callerOnPayload?.(prepared, model);
			return replaced === undefined ? prepared : replaced;
		},
		onProviderStreamEvent: async (event, model) => {
			// Observation only: pi documents this hook's payload as read-only.
			observeUsage(event, observation);
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
			const observation: UsageObservation = { writes: 0 };
			return withVeniceUsage(
				api.stream(model, context, withDefaults(options, layers, thinkingOff(model, level), observation)),
				model,
				observation,
			);
		},
		streamSimple: (model, context, options) => {
			const level = clamp(model, options?.reasoning ?? "off");
			const layers = veniceParameterLayers(model, level, options?.samplingParams);
			const observation: UsageObservation = { writes: 0 };
			return withVeniceUsage(
				api.streamSimple(model, context, withDefaults(options, layers, thinkingOff(model, level), observation)),
				model,
				observation,
			);
		},
	};
}
