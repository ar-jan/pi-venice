/**
 * Venice request defaults, applied to the final request body.
 *
 * Pi merges sampling parameters one level deep: the last of the model's `samplingParams`, the
 * thinking level's override, and the request's `samplingParams` that sets `venice_parameters`
 * replaces it whole. Merging `venice_parameters` here, key by key and after pi has assembled the
 * body, keeps the extension default and every user layer together.
 */

import type {
	Api,
	Model,
	ModelThinkingLevel,
	ProviderEnv,
	ProviderStreams,
	SamplingParams,
	StreamOptions,
} from "@earendil-works/pi-ai";

/** Venice keeps the first 64 characters, like OpenAI. */
const PROMPT_CACHE_KEY_MAX_LENGTH = 64;

/** pi-ai's `clampThinkingLevel`, passed in so this module has no runtime dependencies. */
export type ClampThinkingLevel = (model: Model<Api>, level: ModelThinkingLevel) => ModelThinkingLevel;

export interface VeniceRequestContext {
	sessionId?: string;
	cacheRetention?: StreamOptions["cacheRetention"];
	env?: ProviderEnv;
	/** User `venice_parameters` values in pi's merge order: model, thinking level, request. */
	veniceParameterLayers?: readonly unknown[];
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
	const proto = Object.getPrototypeOf(value);
	return proto === Object.prototype || proto === null;
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
 * - `venice_parameters`: `include_venice_system_prompt: false`, then each user layer key by key,
 *   then the body's own value, so later layers win per key. A non-object value in the body is left
 *   for Venice to reject; non-object layers are skipped.
 * - `prompt_cache_key` from the session ID, so Venice routes the session to servers holding its
 *   cached prefix. Skipped when the body already has one or prompt caching is off.
 */
export function applyVeniceDefaults(payload: unknown, context: VeniceRequestContext): unknown {
	if (!isPlainObject(payload)) return payload;
	let body = payload;

	const veniceParameters = body.venice_parameters;
	if (veniceParameters === undefined || isPlainObject(veniceParameters)) {
		const merged: Record<string, unknown> = { include_venice_system_prompt: false };
		for (const layer of [...(context.veniceParameterLayers ?? []), veniceParameters]) {
			if (isPlainObject(layer)) Object.assign(merged, layer);
		}
		body = { ...body, venice_parameters: merged };
	}

	const cacheKey = body.prompt_cache_key === undefined ? promptCacheKey(context) : undefined;
	if (cacheKey) body = { ...body, prompt_cache_key: cacheKey };

	return body;
}

/** The `venice_parameters` of each layer pi merges, for the clamped thinking level. */
function veniceParameterLayers(
	model: Model<Api>,
	level: ModelThinkingLevel,
	requestParams: SamplingParams | undefined,
	clamp: ClampThinkingLevel,
): unknown[] {
	return [
		model.samplingParams?.venice_parameters,
		model.samplingParamsByThinkingLevel?.[clamp(model, level)]?.venice_parameters,
		requestParams?.venice_parameters,
	];
}

function withDefaults<T extends StreamOptions>(options: T | undefined, layers: readonly unknown[]): T {
	const callerOnPayload = options?.onPayload;
	return {
		...options,
		onPayload: async (payload, model) => {
			const prepared = applyVeniceDefaults(payload, { ...options, veniceParameterLayers: layers });
			// Extension `before_provider_request` hooks arrive here and see the prepared body.
			const replaced = await callerOnPayload?.(prepared, model);
			return replaced === undefined ? prepared : replaced;
		},
	} as T;
}

/** Wrap an API implementation so every request through it gets the Venice defaults. */
export function withVeniceRequestDefaults(api: ProviderStreams, clamp: ClampThinkingLevel): ProviderStreams {
	return {
		...api,
		stream: (model, context, options) => {
			// The completions adapter reads `reasoningEffort`; pi uses "off" when it is unset.
			const level = (options as { reasoningEffort?: ModelThinkingLevel } | undefined)?.reasoningEffort ?? "off";
			const layers = veniceParameterLayers(model, level, options?.samplingParams, clamp);
			return api.stream(model, context, withDefaults(options, layers));
		},
		streamSimple: (model, context, options) => {
			const layers = veniceParameterLayers(model, options?.reasoning ?? "off", options?.samplingParams, clamp);
			return api.streamSimple(model, context, withDefaults(options, layers));
		},
	};
}
