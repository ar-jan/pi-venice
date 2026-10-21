/**
 * Venice request defaults, applied to the final request body.
 *
 * Pi merges sampling parameters one level deep, so a `venice_parameters` object set by the user
 * (models.json overrides, per-thinking-level params, request params) would replace a default kept
 * in `Model.samplingParams`. Merging here, after pi has assembled the body, keeps user keys and the
 * default together.
 */

import type { ProviderEnv, ProviderStreams, StreamOptions } from "@earendil-works/pi-ai";

/** Venice keeps the first 64 characters, like OpenAI. */
const PROMPT_CACHE_KEY_MAX_LENGTH = 64;

export interface VeniceRequestContext {
	sessionId?: string;
	cacheRetention?: StreamOptions["cacheRetention"];
	env?: ProviderEnv;
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
 * - `venice_parameters.include_venice_system_prompt: false` unless the body already sets it.
 *   Other `venice_parameters` keys are kept; a non-object value is left for Venice to reject.
 * - `prompt_cache_key` from the session ID, so Venice routes the session to servers holding its
 *   cached prefix. Skipped when the body already has one or prompt caching is off.
 */
export function applyVeniceDefaults(payload: unknown, context: VeniceRequestContext): unknown {
	if (!isPlainObject(payload)) return payload;
	let body = payload;

	const veniceParameters = body.venice_parameters;
	if (veniceParameters === undefined || isPlainObject(veniceParameters)) {
		body = { ...body, venice_parameters: { include_venice_system_prompt: false, ...veniceParameters } };
	}

	const cacheKey = body.prompt_cache_key === undefined ? promptCacheKey(context) : undefined;
	if (cacheKey) body = { ...body, prompt_cache_key: cacheKey };

	return body;
}

function withDefaults<T extends StreamOptions>(options: T | undefined): T {
	const callerOnPayload = options?.onPayload;
	return {
		...options,
		onPayload: async (payload, model) => {
			const prepared = applyVeniceDefaults(payload, options ?? {});
			// Extension `before_provider_request` hooks arrive here and see the prepared body.
			const replaced = await callerOnPayload?.(prepared, model);
			return replaced === undefined ? prepared : replaced;
		},
	} as T;
}

/** Wrap an API implementation so every request through it gets the Venice defaults. */
export function withVeniceRequestDefaults(api: ProviderStreams): ProviderStreams {
	return {
		...api,
		stream: (model, context, options) => api.stream(model, context, withDefaults(options)),
		streamSimple: (model, context, options) => api.streamSimple(model, context, withDefaults(options)),
	};
}
