/**
 * API-key failover for Venice.
 *
 * Pi stores one credential per provider, so a request carries a single key: the stored
 * `/login venice` credential or `VENICE_API_KEY`. Venice answers a request whose key has no
 * USD/DIEM balance left with HTTP 402 (`INSUFFICIENT_BALANCE`) before any stream content.
 *
 * This wrapper turns that failure into a transparent retry on the next key. Normal requests keep
 * their key and callbacks; only a response with HTTP status 402 switches keys. The status comes from
 * `onResponse`, which pi calls for error responses too since 1.6.1 (earlier versions report only
 * successful ones, so they never rotate). Error text is never matched: balance wording inside a
 * response that already streams, or an unrelated error that mentions 402, leaves the key alone. The
 * switched key is remembered for the process lifetime, so later requests do not hit the exhausted key
 * again.
 *
 * Fallback keys come from numbered variables `VENICE_API_KEY_2`, `VENICE_API_KEY_3`, ... in the
 * process environment or a provider-scoped `env` entry in auth.json. By default the primary key is
 * used first; `/venice-key` can start from any configured key and remembers that choice.
 *
 * A key reaches Venice through `apiKey`, or through an `Authorization: Bearer <key>` header when
 * Pi's `authHeader` setting copies the resolved key into one. That header outranks `apiKey` in the
 * adapter, so it is switched along with the key. A request authenticated by any other
 * `Authorization` value is forwarded as is: switching keys could not change its credential.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { createAssistantMessageEventStream, envApiKeyAuth } from "@earendil-works/pi-ai";
import type {
	Api,
	ApiKeyAuth,
	AssistantMessageEvent,
	AssistantMessageEventStream,
	Model,
	ProviderEnv,
	ProviderHeaders,
	ProviderResponse,
	ProviderStreams,
	StreamOptions,
} from "@earendil-works/pi-ai";
import { getAgentDir, type ModelRegistry } from "@earendil-works/pi-coding-agent";

export const FALLBACK_KEY_PREFIX = "VENICE_API_KEY_";
const FALLBACK_KEY_RE = /^VENICE_API_KEY_(\d+)$/;

/** Indices are 2 and up: index 1 is the primary key's `VENICE_API_KEY`. */
function fallbackIndex(name: string): number | undefined {
	const match = FALLBACK_KEY_RE.exec(name);
	if (!match) return undefined;
	const index = Number(match[1]);
	return index >= 2 ? index : undefined;
}

/**
 * The HTTP status of a request whose key cannot pay for it. Venice documents three codes with it:
 * `INSUFFICIENT_BALANCE`, `API_KEY_DIEM_SPEND_LIMIT_EXCEEDED`, and
 * `API_KEY_USD_SPEND_LIMIT_EXCEEDED`. Any other 402 is Venice's payment path too, so the status
 * alone decides: narrower than "any error" and broad enough for new balance codes.
 */
const PAYMENT_REQUIRED = 402;

/**
 * Fallback keys in index order. A provider-scoped value (auth.json `env`) wins over the process
 * environment for the same index; each value may be a comma- or whitespace-separated list.
 */
function fallbackKeys(env: ProviderEnv | undefined): string[] {
	const byIndex = new Map<number, string>();
	for (const source of [env, process.env]) {
		for (const [name, value] of Object.entries(source ?? {})) {
			const index = fallbackIndex(name);
			if (index !== undefined && typeof value === "string" && !byIndex.has(index)) byIndex.set(index, value);
		}
	}
	return [...byIndex.entries()]
		.sort(([a], [b]) => a - b)
		.flatMap(([, value]) =>
			value
				.split(/[\s,]+/)
				.map((key) => key.trim())
				.filter((key) => key.length > 0),
		);
}

/** The next free `VENICE_API_KEY_N` index, considering auth.json env and the process env. */
export function nextFallbackIndex(env: ProviderEnv | undefined): number {
	const used = new Set<number>();
	for (const source of [env, process.env]) {
		for (const name of Object.keys(source ?? {})) {
			const index = fallbackIndex(name);
			if (index !== undefined) used.add(index);
		}
	}
	let index = 2;
	while (used.has(index)) index++;
	return index;
}

const UNVERIFIED_STORE = "Pi could not verify that the active credential store is empty for Venice. Login stopped.";

/** Pi persists the login result as a replacement. Only the host's store sees the raw current credential. */
export function veniceApiKeyAuth(getRegistry: () => ModelRegistry | undefined): ApiKeyAuth {
	const base = envApiKeyAuth("Venice API key", ["VENICE_API_KEY"]);
	const loginCheck = new AsyncLocalStorage<{ empty: boolean }>();

	async function checkFirstLogin(registry: ModelRegistry): Promise<void> {
		const check = { empty: false };
		const { source } = registry.getProviderAuthStatus("venice");
		// A runtime override can hide a stored entry. A negative status is only a snapshot.
		if (source !== "stored" && source !== "runtime") {
			await loginCheck.run(check, () => registry.getProviderAuth("venice"));
		}
		if (!check.empty) {
			throw new Error(
				"Login can only set the first Venice key. To add a fallback, set the next unused VENICE_API_KEY_N (starting at 2) in the environment or in the active Venice credential's env.",
			);
		}
		if (process.env.VENICE_API_KEY) {
			const name = `${FALLBACK_KEY_PREFIX}${nextFallbackIndex(undefined)}`;
			throw new Error(
				`VENICE_API_KEY is set, so Venice is configured from the environment and a stored key would override it. Add a fallback with: export ${name}=<key>`,
			);
		}
	}

	return {
		...base,
		resolve(input) {
			// An entry with an unresolved $VAR or !command still has a credential object.
			// Observe only its presence, scoped to this login's own registry request.
			const check = loginCheck.getStore();
			if (check) check.empty = input.credential === undefined;
			return base.resolve(input);
		},
		login: async (interaction) => {
			const registry = getRegistry();
			if (!registry) throw new Error(UNVERIFIED_STORE);
			await checkFirstLogin(registry);
			const credential = await base.login!(interaction);
			// Another process can add a credential while the prompt is open.
			await checkFirstLogin(registry);
			interaction.signal.throwIfAborted();
			if (registry !== getRegistry()) throw new Error(UNVERIFIED_STORE);
			return credential;
		},
	};
}

/** Primary key first, then configured fallbacks, without duplicates. */
export function candidateKeys(primary: string | undefined, env: ProviderEnv | undefined): string[] {
	return [...new Set([primary, ...fallbackKeys(env)].filter((key): key is string => Boolean(key)))];
}

/**
 * Which key to use next. State is process-wide because auth resolution returns the primary key for
 * every request; the wrapper has to remember both the user's choice and any failover.
 *
 * `preferred` is the user's selection, persisted across runs. `active` is the key currently in use
 * and may have moved on after a 402. An all-keys-failed request drops `active` back to `preferred`.
 */
export class Keyring {
	private active: string | undefined;
	private preferredHash: string | undefined;

	constructor(preferredHash?: string) {
		this.preferredHash = preferredHash;
	}

	select(keys: readonly string[]): string | undefined {
		if (this.active && keys.includes(this.active)) return this.active;
		if (this.preferredHash) {
			const preferred = keys.find((key) => keyFingerprint(key) === this.preferredHash);
			if (preferred) return (this.active = preferred);
		}
		return (this.active = keys[0]);
	}

	/** Next key after `from`, or undefined when `from` was the last candidate. */
	rotate(keys: readonly string[], from: string): string | undefined {
		const index = keys.indexOf(from);
		if (index < 0 || index + 1 >= keys.length) return undefined;
		return (this.active = keys[index + 1]);
	}

	/** Make `key` the user's choice from now on, including after a restart. */
	choose(key: string): void {
		this.active = key;
		this.preferredHash = keyFingerprint(key);
	}

	/** Forget the current failover, so the next request tries the user's preferred key again. */
	reset(): void {
		this.active = undefined;
	}
}

export function keyFingerprint(key: string): string {
	return createHash("sha256").update(key).digest("hex");
}

/** Short, non-reversible display form used by `/venice-key`. */
export function maskedKey(key: string): string {
	return key.length <= 4 ? "****" : `…${key.slice(-4)}`;
}

/** The env variable a fallback key came from, when it can be determined. */
function keyVariable(key: string, env: ProviderEnv | undefined): string | undefined {
	for (const source of [env, process.env]) {
		for (const [name, value] of Object.entries(source ?? {})) {
			if (fallbackIndex(name) === undefined || typeof value !== "string") continue;
			const keys = value
				.split(/[\s,]+/)
				.map((candidate) => candidate.trim());
			if (keys.includes(key)) return name;
		}
	}
	return undefined;
}

export interface KeyChoice {
	key: string;
	label: string;
}

/** Labels for the selector: number, source variable, masked key, and the current marker. */
export function describeKeys(
	keys: readonly string[],
	env: ProviderEnv | undefined,
	source: string | undefined,
	active: string | undefined,
): KeyChoice[] {
	return keys.map((key, index) => {
		const from = index === 0 ? (source ?? "primary") : (keyVariable(key, env) ?? `fallback ${index + 1}`);
		const marker = key === active ? "●" : " ";
		return { key, label: `${marker} ${index + 1}. ${from} (${maskedKey(key)})` };
	});
}

/** Stable, non-secret way to remember a key across runs. */
const PREFERENCE_FILE = "venice-keys.json";

function preferencePath(): string {
	return join(getAgentDir(), PREFERENCE_FILE);
}

export function loadPreferredKeyHash(): string | undefined {
	try {
		const parsed = JSON.parse(readFileSync(preferencePath(), "utf8")) as { active?: unknown };
		return typeof parsed.active === "string" ? parsed.active : undefined;
	} catch {
		return undefined;
	}
}

export function savePreferredKeyHash(hash: string): void {
	try {
		const path = preferencePath();
		mkdirSync(dirname(path), { recursive: true });
		writeFileSync(path, `${JSON.stringify({ active: hash }, null, 2)}\n`, "utf8");
	} catch {
		// Selection still applies for this process when the preference file is not writable.
	}
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
 * Drive one request, retrying on a different key when Venice rejects it with HTTP 402. `produce`
 * starts an attempt and reports each HTTP status it receives. The adapter reports the status before
 * it streams anything, so a 402 attempt has forwarded no events and can be retried as if it never
 * happened. Every other failure is passed through.
 */
function rotatedAttempt(
	keys: readonly string[],
	keyring: Keyring,
	produce: (apiKey: string, observeStatus: (status: number) => void) => AssistantMessageEventStream,
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
			// The last status wins: the adapter's own retries (429, 5xx) report one status per attempt.
			const response: { status?: number } = {};
			let attempt: AssistantMessageEventStream;
			try {
				attempt = produce(key, (status) => {
					response.status = status;
				});
			} catch (thrown) {
				out.push(errorEvent(thrown instanceof Error ? thrown.message : String(thrown)));
				out.end();
				return;
			}

			let failure: AssistantMessageEvent | undefined;
			let terminal = false;
			try {
				for await (const event of attempt) {
					if (event.type === "error") {
						failure = event;
						break;
					}
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
				const balanceFailure = response.status === PAYMENT_REQUIRED;
				const next = balanceFailure ? keyring.rotate(keys, key) : undefined;
				if (next) {
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

function authorizationHeaders(headers: ProviderHeaders | undefined): [string, string | null][] {
	return Object.entries(headers ?? {}).filter(([name]) => name.toLowerCase() === "authorization");
}

function bearerToken(value: string | null): string | undefined {
	return value === null ? undefined : /^bearer\s+(\S+)\s*$/i.exec(value)?.[1];
}

/**
 * Whether an `Authorization` header (or a `null` that suppresses it) decides the credential instead
 * of the resolved key. Switching `apiKey` would resend that same credential.
 */
function authorizedOutsideKey(options: StreamOptions | undefined): boolean {
	return authorizationHeaders(options?.headers).some(([, value]) => bearerToken(value) !== options?.apiKey);
}

/** Options that authenticate with `apiKey`, including a header Pi's `authHeader` built from the resolved key. */
function withApiKey<T extends StreamOptions>(options: T | undefined, apiKey: string): T {
	const headers = options?.headers;
	if (!headers || apiKey === options?.apiKey || authorizationHeaders(headers).length === 0) {
		return { ...options, apiKey } as T;
	}
	const switched = Object.fromEntries(
		Object.entries(headers).map(([name, value]) => [
			name,
			name.toLowerCase() === "authorization" ? `Bearer ${apiKey}` : value,
		]),
	);
	return { ...options, apiKey, headers: switched } as T;
}

/**
 * Options for one attempt on `apiKey` that report each HTTP status to `observeStatus`, then to the
 * caller's `onResponse`, so pi still sees every attempt's response, including a rotated-away 402.
 */
function attemptOptions<T extends StreamOptions>(
	options: T | undefined,
	apiKey: string,
	observeStatus: (status: number) => void,
): T {
	const callerOnResponse = options?.onResponse;
	return {
		...withApiKey(options, apiKey),
		onResponse: async (response: ProviderResponse, model: Model<Api>) => {
			observeStatus(response.status);
			await callerOnResponse?.(response, model);
		},
	} as T;
}

/** Wrap a provider's streams so an HTTP 402 on one key transparently retries the next key. */
export function withKeyRotation(api: ProviderStreams, keyring: Keyring = new Keyring()): ProviderStreams {
	return {
		...api,
		stream: (model, context, options) => {
			if (authorizedOutsideKey(options)) return api.stream(model, context, options);
			const keys = candidateKeys(options?.apiKey, options?.env);
			return rotatedAttempt(keys, keyring, (apiKey, observeStatus) =>
				api.stream(model, context, attemptOptions(options, apiKey, observeStatus)),
			);
		},
		streamSimple: (model, context, options) => {
			if (authorizedOutsideKey(options)) return api.streamSimple(model, context, options);
			const keys = candidateKeys(options?.apiKey, options?.env);
			return rotatedAttempt(keys, keyring, (apiKey, observeStatus) =>
				api.streamSimple(model, context, attemptOptions(options, apiKey, observeStatus)),
			);
		},
	};
}
