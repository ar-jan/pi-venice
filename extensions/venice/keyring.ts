/**
 * API-key failover for Venice.
 *
 * Pi stores one credential per provider, so a request carries a single key: the stored
 * `/login venice` credential or `VENICE_API_KEY`. Venice answers a request whose key has no
 * USD/DIEM balance left with HTTP 402 (`INSUFFICIENT_BALANCE`) before any stream content.
 *
 * This wrapper turns that failure into a transparent retry on the next key. Normal requests are
 * forwarded unchanged; only a 402 before the response starts switches keys. The switched key is
 * remembered for the process lifetime, so later requests do not hit the exhausted key again.
 *
 * Fallback keys come from `VENICE_API_KEY_2` (comma- or whitespace-separated), either from the
 * process environment or from a provider-scoped `env` entry in auth.json. The primary key always
 * comes first, so key1 is used until Venice reports it exhausted.
 */

import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import type {
	AssistantMessageEvent,
	AssistantMessageEventStream,
	ProviderEnv,
	ProviderStreams,
} from "@earendil-works/pi-ai";

export const FALLBACK_KEY_ENV = "VENICE_API_KEY_2";

/**
 * Venice documents these 402 codes for a key that cannot pay for the request:
 * `INSUFFICIENT_BALANCE`, `API_KEY_DIEM_SPEND_LIMIT_EXCEEDED`, and
 * `API_KEY_USD_SPEND_LIMIT_EXCEEDED`. Any other 402 is Venice's payment path too, so matching the
 * status is both narrower than "any error" and broad enough for new balance codes.
 */
const BALANCE_FAILURE = /\b402\b|INSUFFICIENT_BALANCE|SPEND_LIMIT_EXCEEDED/i;

function fallbackKeys(env: ProviderEnv | undefined): string[] {
	const raw = env?.[FALLBACK_KEY_ENV] ?? process.env[FALLBACK_KEY_ENV] ?? "";
	return raw
		.split(/[\s,]+/)
		.map((key) => key.trim())
		.filter((key) => key.length > 0);
}

/** Primary key first, then configured fallbacks, without duplicates. */
export function candidateKeys(primary: string | undefined, env: ProviderEnv | undefined): string[] {
	return [...new Set([primary, ...fallbackKeys(env)].filter((key): key is string => Boolean(key)))];
}

/**
 * Which key to use next. State is process-wide because auth resolution returns the primary key for
 * every request; the wrapper has to remember that the primary ran dry.
 */
export class Keyring {
	private active: string | undefined;

	select(keys: readonly string[]): string | undefined {
		if (this.active && keys.includes(this.active)) return this.active;
		return (this.active = keys[0]);
	}

	/** Next key after `from`, or undefined when `from` was the last candidate. */
	rotate(keys: readonly string[], from: string): string | undefined {
		const index = keys.indexOf(from);
		if (index < 0 || index + 1 >= keys.length) return undefined;
		return (this.active = keys[index + 1]);
	}

	/** Forget the switch, so the next request tries the primary key again. */
	reset(): void {
		this.active = undefined;
	}
}

function isBalanceFailure(event: AssistantMessageEvent): boolean {
	return event.type === "error" && BALANCE_FAILURE.test(event.error.errorMessage ?? "");
}

function errorEvent(message: string): AssistantMessageEvent {
	return {
		type: "error",
		reason: "error",
		error: {
			role: "assistant",
			content: [],
			api: "openai-completions",
			provider: "venice",
			model: "unknown",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "error",
			errorMessage: message,
			timestamp: Date.now(),
		},
	};
}

/**
 * Drive one request, retrying on a different key when Venice rejects it for insufficient balance.
 * A retry is only possible before the response started; once content has been forwarded, a late
 * failure is passed through like any other error.
 */
function rotatedAttempt(
	keys: readonly string[],
	keyring: Keyring,
	produce: (apiKey: string) => AssistantMessageEventStream,
): AssistantMessageEventStream {
	const out = createAssistantMessageEventStream();

	const first = keyring.select(keys);
	if (!first) {
		// Pi only calls a provider with a resolved key; keep the failure explicit if that changes.
		out.push(errorEvent("No Venice API key is configured"));
		out.end();
		return out;
	}

	void (async () => {
		let index = keys.indexOf(first);
		for (;;) {
			const key = keys[index]!;
			let attempt: AssistantMessageEventStream;
			try {
				attempt = produce(key);
			} catch (thrown) {
				out.push(errorEvent(thrown instanceof Error ? thrown.message : String(thrown)));
				out.end();
				return;
			}

			let started = false;
			let failure: AssistantMessageEvent | undefined;
			let terminal = false;
			try {
				for await (const event of attempt) {
					if (event.type === "error") {
						failure = event;
						break;
					}
					if (event.type === "start") started = true;
					out.push(event);
					if (event.type === "done") {
						terminal = true;
						break;
					}
				}
			} catch (thrown) {
				// The inner stream normally reports errors as events; treat a thrown iterator as one.
				failure = errorEvent(thrown instanceof Error ? thrown.message : String(thrown));
			}

			if (terminal) {
				out.end();
				return;
			}

			if (failure) {
				const balanceFailure = isBalanceFailure(failure);
				const next = balanceFailure ? keyring.rotate(keys, key) : undefined;
				if (next && !started) {
					index = keys.indexOf(next);
					continue;
				}
				// Every candidate failed: let the next request try the primary key again.
				if (balanceFailure && !next) keyring.reset();
				out.push(failure);
				out.end();
				return;
			}

			// A stream that ended without `done` or `error` cannot produce a result.
			out.end();
			return;
		}
	})();

	return out;
}

/** Wrap a provider's streams so a balance failure on one key transparently retries the next key. */
export function withKeyRotation(api: ProviderStreams): ProviderStreams {
	const keyring = new Keyring();
	return {
		...api,
		stream: (model, context, options) => {
			const keys = candidateKeys(options?.apiKey, options?.env);
			return rotatedAttempt(keys, keyring, (apiKey) => api.stream(model, context, { ...options, apiKey }));
		},
		streamSimple: (model, context, options) => {
			const keys = candidateKeys(options?.apiKey, options?.env);
			return rotatedAttempt(keys, keyring, (apiKey) => api.streamSimple(model, context, { ...options, apiKey }));
		},
	};
}
