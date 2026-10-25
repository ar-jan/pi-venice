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
 * Fallback keys come from numbered variables `VENICE_API_KEY_2`, `VENICE_API_KEY_3`, ... in the
 * process environment or a provider-scoped `env` entry in auth.json. By default the primary key is
 * used first; `/venice-key` can start from any configured key and remembers that choice.
 */

import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { createAssistantMessageEventStream, envApiKeyAuth } from "@earendil-works/pi-ai";
import type {
	ApiKeyAuth,
	ApiKeyCredential,
	AssistantMessageEvent,
	AssistantMessageEventStream,
	ProviderEnv,
	ProviderStreams,
} from "@earendil-works/pi-ai";
import { getAgentDir } from "@earendil-works/pi-coding-agent";

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
 * Venice documents these 402 codes for a key that cannot pay for the request:
 * `INSUFFICIENT_BALANCE`, `API_KEY_DIEM_SPEND_LIMIT_EXCEEDED`, and
 * `API_KEY_USD_SPEND_LIMIT_EXCEEDED`. Any other 402 is Venice's payment path too, so matching the
 * status is both narrower than "any error" and broad enough for new balance codes.
 */
const BALANCE_FAILURE = /\b402\b|INSUFFICIENT_BALANCE|SPEND_LIMIT_EXCEEDED/i;

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

/** Read the credential `/login venice` previously stored, if any. */
function readStoredCredential(): ApiKeyCredential | undefined {
	try {
		const parsed = JSON.parse(readFileSync(join(getAgentDir(), "auth.json"), "utf8")) as { venice?: ApiKeyCredential };
		const credential = parsed.venice;
		if (credential?.type === "api_key" && credential.key) return credential;
	} catch {
		// No readable auth.json: first-time login, or a store this extension cannot inspect.
	}
	return undefined;
}

/**
 * Venice API-key auth whose login adds keys instead of replacing them.
 *
 * The first `/login venice` stores the primary key. Later runs append the entered key to the stored
 * credential's `env` as the next `VENICE_API_KEY_N`, which the request wrapper uses as a fallback.
 * When the primary comes from `VENICE_API_KEY` instead, a stored key would silently override that
 * environment variable, so login explains the environment variable to set instead.
 */
export function veniceApiKeyAuth(): ApiKeyAuth {
	const base = envApiKeyAuth("Venice API key", ["VENICE_API_KEY"]);
	return {
		...base,
		login: async (interaction) => {
			const stored = readStoredCredential();
			if (stored) {
				const name = `${FALLBACK_KEY_PREFIX}${nextFallbackIndex(stored.env)}`;
				const key = await interaction.prompt({
					type: "secret",
					message: `Enter the fallback Venice API key to store as ${name}`,
				});
				interaction.signal.throwIfAborted();
				interaction.notify({
					type: "info",
					message: `Stored as ${name} in auth.json. It is used after the stored key runs out; run /login venice again to add another key.`,
				});
				return { ...stored, env: { ...stored.env, [name]: key } };
			}
			if (process.env.VENICE_API_KEY) {
				const name = `${FALLBACK_KEY_PREFIX}${nextFallbackIndex(undefined)}`;
				throw new Error(
					`VENICE_API_KEY is set, so Venice is configured from the environment and a stored key would override it. Add a fallback with: export ${name}=<key>`,
				);
			}
			const key = await interaction.prompt({ type: "secret", message: "Enter Venice API key" });
			interaction.signal.throwIfAborted();
			return { type: "api_key", key };
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
export function withKeyRotation(api: ProviderStreams, keyring: Keyring = new Keyring()): ProviderStreams {
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
