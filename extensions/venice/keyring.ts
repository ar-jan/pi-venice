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
	ApiKeyCredential,
	AssistantMessageEvent,
	AssistantMessageEventStream,
	ClassifierApi,
	ClassifierModel,
	ClassifierOptions,
	ClassifierResult,
	Model,
	ProviderClassifier,
	ProviderEnv,
	ProviderHeaders,
	ProviderRequestOptions,
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

const UNVERIFIED_STORE = "Pi could not verify the Venice entry in the active credential store. Login stopped.";

/** What a login check saw of the active store's Venice entry, through this login's own resolve. */
interface LoginCheck {
	observed: boolean;
	stored?: ApiKeyCredential;
}

/**
 * Pi persists the login result as a replacement. Only the host's store sees the raw current
 * credential, so login refuses to replace one that works. A stored key that does not resolve (a
 * failing `!command` or an unset `$VAR`) can be entered again: its `env`, which holds the fallback
 * keys, is copied onto the new credential.
 */
export function veniceApiKeyAuth(getRegistry: () => ModelRegistry | undefined): ApiKeyAuth {
	const base = envApiKeyAuth("Venice API key", ["VENICE_API_KEY"]);
	const loginCheck = new AsyncLocalStorage<LoginCheck>();

	/** The env of the stored entry that login replaces, or undefined when nothing is stored. */
	async function replaceableEnv(registry: ModelRegistry): Promise<ProviderEnv | undefined> {
		const check: LoginCheck = { observed: false };
		// A runtime override hides the stored entry from resolution. Any other status is only a snapshot.
		const resolved = registry.getProviderAuthStatus("venice").source === "runtime"
			? undefined
			: await loginCheck.run(check, () => registry.getProviderAuth("venice"));
		if (!check.observed || check.stored?.key) {
			throw new Error(
				"Login can only set the first Venice key. To add a fallback, set the next unused VENICE_API_KEY_N (starting at 2) in the environment or in the active Venice credential's env.",
			);
		}
		// Without a usable stored key, the resolver can still supply VENICE_API_KEY.
		if (resolved?.auth.apiKey) {
			const name = `${FALLBACK_KEY_PREFIX}${nextFallbackIndex(check.stored?.env)}`;
			throw new Error(
				`VENICE_API_KEY is set, so Venice is configured from the environment and a stored key would override it. Add a fallback with: export ${name}=<key>`,
			);
		}
		return check.stored?.env;
	}

	return {
		...base,
		resolve(input) {
			// The store hands over the entry with its key resolved: undefined when a $VAR or !command
			// does not resolve, while its env stays as written. Observed only for this login's request.
			const check = loginCheck.getStore();
			if (check) {
				check.observed = true;
				check.stored = input.credential;
			}
			return base.resolve(input);
		},
		login: async (interaction) => {
			const registry = getRegistry();
			if (!registry) throw new Error(UNVERIFIED_STORE);
			await replaceableEnv(registry);
			const credential = await base.login!(interaction);
			if (!credential.key?.trim()) throw new Error("Venice API key cannot be blank.");
			// Another process can change the entry while the prompt is open. Keep the env it has now.
			const env = await replaceableEnv(registry);
			interaction.signal.throwIfAborted();
			if (registry !== getRegistry()) throw new Error(UNVERIFIED_STORE);
			return env ? { ...credential, env } : credential;
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

type StreamErrorEvent = Extract<AssistantMessageEvent, { type: "error" }>;

function errorEvent(message: string): StreamErrorEvent {
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

interface TrackedAttempt<T> {
	options: T;
	lastStatus(): number | undefined;
}

type KeyedOptions = { apiKey?: string; headers?: ProviderHeaders };

/** Build one attempt's options for `key`, tracking the HTTP status of its final completed response. */
function trackedAttemptOptions<TModel, T extends ProviderRequestOptions<TModel>>(
	options: T | undefined,
	key: string,
	baseFetch: typeof globalThis.fetch,
): TrackedAttempt<T> {
	let lastStatus: number | undefined;
	return {
		options: {
			...withApiKey(options, key),
			fetch: (input, init) => {
				// A transport failure on Pi's retry has no onResponse callback.
				lastStatus = undefined;
				return baseFetch(input, init);
			},
			onResponse: async (response, model) => {
				const { status } = response;
				await options?.onResponse?.(response, model);
				lastStatus = status;
			},
		} as T,
		lastStatus: () => lastStatus,
	};
}

/** Advance to the next candidate key after a 402 error, or reset when every key failed. */
function nextKeyOnPaymentRequired(
	keyring: Keyring,
	keys: readonly string[],
	key: string,
	status: number | undefined,
	isError: boolean,
	signal: AbortSignal | undefined,
): string | undefined {
	if (status !== PAYMENT_REQUIRED || !isError || signal?.aborted) return undefined;
	const next = keyring.rotate(keys, key);
	if (!next) keyring.reset();
	return next;
}

/**
 * Drive one request, retrying on a different key when its final HTTP attempt fails with a 402.
 * The adapter reports that response before forwarding any stream events.
 */
function rotatedAttempt<T extends StreamOptions>(
	options: T | undefined,
	keyring: Keyring,
	produce: (options: T | undefined) => AssistantMessageEventStream,
): AssistantMessageEventStream {
	if (authorizedOutsideKey(options)) return produce(options);

	const out = createAssistantMessageEventStream();
	const keys = candidateKeys(options?.apiKey, options?.env);
	const first = keyring.select(keys);
	if (!first) {
		// Pi only calls a provider with a resolved key; keep the failure explicit if that changes.
		out.push(errorEvent("No Venice API key is configured"));
		out.end();
		return out;
	}

	const baseFetch = options?.fetch ?? globalThis.fetch;
	void (async () => {
		let key = first;
		for (;;) {
			const tracked = trackedAttemptOptions<Model<Api>, T>(options, key, baseFetch);
			let attempt: AssistantMessageEventStream;
			try {
				attempt = produce(tracked.options);
			} catch (thrown) {
				out.push(errorEvent(thrown instanceof Error ? thrown.message : String(thrown)));
				out.end();
				return;
			}

			let failure: StreamErrorEvent | undefined;
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
				const next = nextKeyOnPaymentRequired(
					keyring,
					keys,
					key,
					tracked.lastStatus(),
					failure.reason === "error",
					options?.signal,
				);
				if (next) {
					key = next;
					continue;
				}
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

function classifierError(model: ClassifierModel<ClassifierApi>, message: string): ClassifierResult {
	return {
		api: model.api,
		provider: model.provider,
		model: model.id,
		answers: {},
		stopReason: "error",
		errorMessage: message,
		timestamp: Date.now(),
	};
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
function authorizedOutsideKey(options: KeyedOptions | undefined): boolean {
	return authorizationHeaders(options?.headers).some(([, value]) => bearerToken(value) !== options?.apiKey);
}

/** Options that authenticate with `apiKey`, including a header Pi's `authHeader` built from the resolved key. */
function withApiKey<T extends KeyedOptions>(options: T | undefined, apiKey: string): T {
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

/** Wrap a provider's streams so an HTTP 402 on one key transparently retries the next key. */
export function withKeyRotation(api: ProviderStreams, keyring: Keyring = new Keyring()): ProviderStreams {
	return {
		...api,
		stream: (model, context, options) =>
			rotatedAttempt(options, keyring, (attempt) => api.stream(model, context, attempt)),
		streamSimple: (model, context, options) =>
			rotatedAttempt(options, keyring, (attempt) => api.streamSimple(model, context, attempt)),
	};
}

/** Wrap a classifier so an HTTP 402 on one key transparently retries the next key. */
export function withClassifierKeyRotation(
	classifier: ProviderClassifier,
	keyring: Keyring = new Keyring(),
): ProviderClassifier {
	return {
		classify: async (model, context, options) => {
			if (authorizedOutsideKey(options)) return classifier.classify(model, context, options);

			const keys = candidateKeys(options?.apiKey, options?.env);
			const first = keyring.select(keys);
			if (!first) return classifierError(model, "No Venice API key is configured");

			const baseFetch = options?.fetch ?? globalThis.fetch;
			let key = first;
			for (;;) {
				const tracked = trackedAttemptOptions<ClassifierModel<ClassifierApi>, ClassifierOptions>(
					options,
					key,
					baseFetch,
				);
				let result: ClassifierResult;
				try {
					result = await classifier.classify(model, context, tracked.options);
				} catch (thrown) {
					return classifierError(model, thrown instanceof Error ? thrown.message : String(thrown));
				}

				if (result.stopReason === "stop") return result;

				const next = nextKeyOnPaymentRequired(
					keyring,
					keys,
					key,
					tracked.lastStatus(),
					result.stopReason === "error",
					options?.signal,
				);
				if (!next) return result;
				key = next;
			}
		},
	};
}
