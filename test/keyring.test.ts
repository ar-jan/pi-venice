import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, it, type TestContext } from "node:test";
import { createAssistantMessageEventStream, InMemoryCredentialStore } from "@earendil-works/pi-ai";
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager } from "@earendil-works/pi-coding-agent";
import registerVenice from "../extensions/venice/index.ts";
import type {
	ApiKeyCredential,
	AssistantMessage,
	AssistantMessageEventStream,
	Model,
	ProviderAuthInteraction,
	ProviderHeaders,
	ProviderStreams,
	SimpleStreamOptions,
	StreamOptions,
} from "@earendil-works/pi-ai";
import {
	candidateKeys,
	describeKeys,
	Keyring,
	keyFingerprint,
	loadPreferredKeyHash,
	nextFallbackIndex,
	savePreferredKeyHash,
	veniceApiKeyAuth,
	withKeyRotation,
} from "../extensions/venice/keyring.ts";

const zeroUsage = () => ({
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});

const assistant = (text: string): AssistantMessage => ({
	role: "assistant",
	content: [{ type: "text", text }],
	api: "openai-completions",
	provider: "venice",
	model: "kimi-k3",
	usage: zeroUsage(),
	stopReason: "stop",
	timestamp: Date.now(),
});

/**
 * One fake attempt: its events, and the HTTP status it reports, or one per response when the adapter
 * retried the request itself (none when no response arrived).
 */
interface Outcome {
	kind: "done" | "started-error" | "error";
	message: string;
	status?: number | number[];
}

const outcome = (kind: Outcome["kind"], message: string, status?: Outcome["status"]): Outcome => ({
	kind,
	message,
	status,
});

/** Venice's 402 for a key without balance, as pi's adapter reports it. */
const paymentRequired = () =>
	outcome("error", "402 \"Insufficient USD or Diem balance to complete request.\"", 402);

/** Play an outcome the way pi's adapters do: each HTTP status reaches `onResponse` before any event. */
function play({ kind, message, status }: Outcome, options: StreamOptions | undefined): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();
	const result = assistant(message);
	void (async () => {
		for (const each of [status ?? []].flat()) await options?.onResponse?.({ status: each, headers: {} }, model);
		if (kind === "done") {
			stream.push({ type: "start", partial: result });
			stream.push({ type: "done", reason: "stop", message: result });
		} else {
			if (kind === "started-error") stream.push({ type: "start", partial: result });
			stream.push({
				type: "error",
				reason: "error",
				error: { ...result, stopReason: "error", errorMessage: message },
			});
		}
		stream.end();
	})();
	return stream;
}

/** Fake provider that records the key and headers each attempt used. */
function fakeApi(
	respond: (apiKey: string | undefined) => Outcome,
): ProviderStreams & { calls: (string | undefined)[]; headers: (ProviderHeaders | undefined)[] } {
	const calls: (string | undefined)[] = [];
	const headers: (ProviderHeaders | undefined)[] = [];
	const produce = (options: StreamOptions | undefined) => {
		calls.push(options?.apiKey);
		headers.push(options?.headers);
		return play(respond(options?.apiKey), options);
	};
	return {
		calls,
		headers,
		stream: (_model, _context, options) => produce(options),
		streamSimple: (_model, _context, options) => produce(options),
	};
}

const model: Model<"openai-completions"> = {
	id: "kimi-k3",
	name: "Kimi K3",
	api: "openai-completions",
	provider: "venice",
	baseUrl: "https://api.venice.ai/api/v1",
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128000,
	maxTokens: 16384,
	reasoning: false,
};

const options = (apiKey: string, env?: Record<string, string>) => ({ apiKey, env }) as StreamOptions;

const call = (api: ProviderStreams, streamOptions: StreamOptions) =>
	api.stream(model, {} as never, streamOptions);

const simpleCall = (api: ProviderStreams, streamOptions: SimpleStreamOptions) =>
	api.streamSimple(model, {} as never, streamOptions);

async function drain(stream: AssistantMessageEventStream): Promise<AssistantMessage> {
	for await (const _event of stream) {
		// drain
	}
	return await stream.result();
}

describe("candidateKeys", () => {
	afterEach(() => {
		delete process.env.VENICE_API_KEY_2;
		delete process.env.VENICE_API_KEY_3;
	});

	it("uses the primary key alone when no fallback is configured", () => {
		assert.deepEqual(candidateKeys("key-1", undefined), ["key-1"]);
	});

	it("appends provider-scoped fallbacks and removes duplicates", () => {
		assert.deepEqual(candidateKeys("key-1", { VENICE_API_KEY_2: "key-2, key-3 key-1" }), [
			"key-1",
			"key-2",
			"key-3",
		]);
	});

	it("orders numbered fallbacks and lets auth.json env win per index", () => {
		process.env.VENICE_API_KEY_3 = "process-3";
		process.env.VENICE_API_KEY_2 = "process-2";
		assert.deepEqual(candidateKeys("key-1", { VENICE_API_KEY_2: "stored-2" }), [
			"key-1",
			"stored-2",
			"process-3",
		]);
	});

	it("falls back to the process environment", () => {
		process.env.VENICE_API_KEY_2 = "key-2";
		assert.deepEqual(candidateKeys("key-1", undefined), ["key-1", "key-2"]);
	});
});

describe("withKeyRotation", () => {
	const fallback = { VENICE_API_KEY_2: "key-2" };

	it("retries on the fallback key when Venice answers HTTP 402", async () => {
		const api = fakeApi((key) => (key === "key-1" ? paymentRequired() : outcome("done", "ok", 200)));
		const rotated = withKeyRotation(api);
		const result = await drain(call(rotated, options("key-1", fallback)));

		assert.deepEqual(api.calls, ["key-1", "key-2"]);
		assert.equal(result.stopReason, "stop");
		assert.equal(result.content[0]?.type === "text" && result.content[0].text, "ok");
	});

	it("keeps using the fallback key for later requests", async () => {
		const api = fakeApi((key) => (key === "key-1" ? paymentRequired() : outcome("done", "ok", 200)));
		const rotated = withKeyRotation(api);

		await drain(call(rotated, options("key-1", fallback)));
		await drain(call(rotated, options("key-1", fallback)));

		assert.deepEqual(api.calls, ["key-1", "key-2", "key-2"]);
	});

	it("decides on the last status when the adapter retried the request itself", async () => {
		const api = fakeApi((key) => {
			if (key === "key-1") return { ...paymentRequired(), status: [429, 402] };
			// A rate limit that the adapter retried until it gave up is not a balance failure.
			return outcome("error", "429 Too Many Requests", [402, 429]);
		});
		const rotated = withKeyRotation(api);
		const result = await drain(call(rotated, options("key-1", fallback)));

		assert.deepEqual(api.calls, ["key-1", "key-2"]);
		assert.match(result.errorMessage ?? "", /^429\b/);
	});

	// Only the HTTP status counts, never the error text.
	for (const [name, failure] of [
		["balance wording inside a response that already streams", outcome("started-error", "402 INSUFFICIENT_BALANCE", 200)],
		["an unrelated error that mentions 402", outcome("error", "400 max_completion_tokens must be at most 402", 400)],
		["an error without an HTTP response", outcome("error", "402: insufficient balance")],
	] as const) {
		it(`does not switch keys on ${name}`, async () => {
			const api = fakeApi(() => failure);
			const rotated = withKeyRotation(api);

			const result = await drain(call(rotated, options("key-1", fallback)));
			await drain(call(rotated, options("key-1", fallback)));

			assert.deepEqual(api.calls, ["key-1", "key-1"]);
			assert.equal(result.stopReason, "error");
		});
	}

	it("does not retry unrelated errors", async () => {
		const api = fakeApi(() => outcome("error", "500: internal server error", 500));
		const rotated = withKeyRotation(api);
		const result = await drain(call(rotated, options("key-1", fallback)));

		assert.deepEqual(api.calls, ["key-1"]);
		assert.equal(result.stopReason, "error");
	});

	it("passes the balance error through when no fallback is configured", async () => {
		const api = fakeApi(() => paymentRequired());
		const rotated = withKeyRotation(api);
		const result = await drain(call(rotated, options("key-1")));

		assert.deepEqual(api.calls, ["key-1"]);
		assert.match(result.errorMessage ?? "", /^402\b/);
	});

	it("retries the primary key on a later request when every key failed", async () => {
		const api = fakeApi(() => paymentRequired());
		const rotated = withKeyRotation(api);

		await drain(call(rotated, options("key-1", fallback)));
		await drain(call(rotated, options("key-1", fallback)));

		assert.deepEqual(api.calls, ["key-1", "key-2", "key-1", "key-2"]);
	});

	it("rotates streamSimple requests too", async () => {
		const api = fakeApi((key) => (key === "key-1" ? paymentRequired() : outcome("done", "ok", 200)));
		const rotated = withKeyRotation(api);
		const result = await drain(simpleCall(rotated, { apiKey: "key-1", env: fallback }));

		assert.deepEqual(api.calls, ["key-1", "key-2"]);
		assert.equal(result.stopReason, "stop");
	});

	it("still reports every attempt's response to the caller's onResponse", async () => {
		const api = fakeApi((key) => (key === "key-1" ? paymentRequired() : outcome("done", "ok", 200)));
		const statuses: number[] = [];
		await drain(
			call(withKeyRotation(api), {
				...options("key-1", fallback),
				onResponse: (response) => {
					statuses.push(response.status);
				},
			}),
		);

		assert.deepEqual(statuses, [402, 200]);
	});

	describe("with an Authorization header", () => {
		const withHeaders = (apiKey: string, headers: ProviderHeaders): StreamOptions => ({
			apiKey,
			env: fallback,
			headers,
		});
		const balanceFailureOnKey1 = (key: string | undefined) =>
			key === "key-1" ? paymentRequired() : outcome("done", "ok", 200);

		it("switches the header Pi's authHeader built from the resolved key", async () => {
			const api = fakeApi(balanceFailureOnKey1);
			const rotated = withKeyRotation(api);
			const result = await drain(call(rotated, withHeaders("key-1", { Authorization: "Bearer key-1", "X-Trace": "t" })));

			assert.deepEqual(api.calls, ["key-1", "key-2"]);
			assert.deepEqual(api.headers, [
				{ Authorization: "Bearer key-1", "X-Trace": "t" },
				{ Authorization: "Bearer key-2", "X-Trace": "t" },
			]);
			assert.equal(result.stopReason, "stop");
		});

		it("sends a chosen key in the header from the first attempt", async () => {
			const keyring = new Keyring();
			keyring.choose("key-2");
			const api = fakeApi(() => outcome("done", "ok"));

			await drain(call(withKeyRotation(api, keyring), withHeaders("key-1", { Authorization: "Bearer key-1" })));

			assert.deepEqual(api.headers, [{ Authorization: "Bearer key-2" }]);
		});

		it("matches the header name and scheme in any case", async () => {
			const api = fakeApi(balanceFailureOnKey1);

			await drain(call(withKeyRotation(api), withHeaders("key-1", { authorization: "bearer key-1" })));

			assert.deepEqual(api.headers.at(-1), { authorization: "Bearer key-2" });
		});

		for (const value of ["Bearer proxy-token", "Basic dXNlcjpwYXNz", null]) {
			it(`forwards a request authorized by ${JSON.stringify(value)} without rotating`, async () => {
				const keyring = new Keyring();
				keyring.choose("key-2");
				const api = fakeApi(() => paymentRequired());
				const rotated = withKeyRotation(api, keyring);

				const result = await drain(call(rotated, withHeaders("key-1", { Authorization: value })));
				await drain(call(rotated, withHeaders("key-1", { Authorization: value })));

				// Switching apiKey would resend the same credential, so neither request is retried.
				assert.deepEqual(api.calls, ["key-1", "key-1"]);
				assert.deepEqual(api.headers, [{ Authorization: value }, { Authorization: value }]);
				assert.equal(result.stopReason, "error");
			});
		}
	});
});

describe("Keyring", () => {
	const keys = ["key-1", "key-2", "key-3"];

	it("starts from the persisted selection", () => {
		assert.equal(new Keyring(keyFingerprint("key-2")).select(keys), "key-2");
	});

	it("falls back to the primary when the persisted key is gone", () => {
		assert.equal(new Keyring(keyFingerprint("removed")).select(keys), "key-1");
	});

	it("keeps a user choice across a reset", () => {
		const keyring = new Keyring();
		keyring.choose("key-3");
		keyring.reset();
		assert.equal(keyring.select(keys), "key-3");
	});

	it("still prefers the user choice after failover", () => {
		const keyring = new Keyring();
		keyring.choose("key-2");
		assert.equal(keyring.select(keys), "key-2");
		assert.equal(keyring.rotate(keys, "key-2"), "key-3");
		assert.equal(keyring.select(keys), "key-3");
		keyring.reset();
		assert.equal(keyring.select(keys), "key-2");
	});
});

describe("describeKeys", () => {
	it("shows the source variable, a masked key, and the current marker", () => {
		const choices = describeKeys(["key-1", "key-2"], { VENICE_API_KEY_2: "key-2" }, "stored credential", "key-2");

		assert.deepEqual(
			choices.map((choice) => choice.label),
			["  1. stored credential (…ey-1)", "● 2. VENICE_API_KEY_2 (…ey-2)"],
		);
	});
});

describe("preferred key persistence", () => {
	const savedAgentDir = process.env.PI_CODING_AGENT_DIR;

	afterEach(() => {
		if (savedAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = savedAgentDir;
	});

	it("round-trips the hash and ignores a missing file", () => {
		const dir = mkdtempSync(join(tmpdir(), "pi-venice-pref-"));
		process.env.PI_CODING_AGENT_DIR = dir;
		try {
			assert.equal(loadPreferredKeyHash(), undefined);
			const hash = keyFingerprint("key-2");
			savePreferredKeyHash(hash);
			assert.equal(loadPreferredKeyHash(), hash);
		} finally {
			rmSync(dir, { recursive: true, force: true });
		}
	});
});

describe("Venice login through Pi's credential store", () => {
	const savedAgentDir = process.env.PI_CODING_AGENT_DIR;
	const savedPrimary = process.env.VENICE_API_KEY;
	const tempDirs: string[] = [];
	const a = { type: "api_key", key: "a-1", env: { VENICE_API_KEY_2: "a-2" } } as const;
	const b = { type: "api_key", key: "b-1", env: { VENICE_API_KEY_2: "b-2" } } as const;
	let defaultDir: string;
	let defaultAuthPath: string;

	function profile(venice?: ApiKeyCredential): string {
		const dir = mkdtempSync(join(tmpdir(), "pi-venice-auth-"));
		tempDirs.push(dir);
		writeFileSync(join(dir, "auth.json"), JSON.stringify({ venice }, null, 2));
		return dir;
	}

	async function veniceRuntime(t: TestContext, agentDir?: string, modelRuntime?: ModelRuntime): Promise<ModelRuntime> {
		const cwd = agentDir ?? defaultDir;
		const resourceLoader = new DefaultResourceLoader({ cwd, agentDir: cwd, extensionFactories: [registerVenice] });
		await resourceLoader.reload();
		const { session } = await createAgentSession({
			cwd,
			agentDir,
			modelRuntime,
			resourceLoader,
			sessionManager: SessionManager.inMemory(cwd),
		});
		t.after(() => session.dispose());
		await session.bindExtensions({});
		return session.modelRuntime;
	}

	function interaction(answer?: string): ProviderAuthInteraction {
		return {
			signal: new AbortController().signal,
			prompt: async () => answer ?? assert.fail("Unexpected login prompt"),
			notify() {},
		};
	}

	beforeEach(() => {
		defaultDir = profile(a);
		defaultAuthPath = join(defaultDir, "auth.json");
		process.env.PI_CODING_AGENT_DIR = defaultDir;
		delete process.env.VENICE_API_KEY;
	});

	afterEach(() => {
		for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
		if (savedAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = savedAgentDir;
		if (savedPrimary === undefined) delete process.env.VENICE_API_KEY;
		else process.env.VENICE_API_KEY = savedPrimary;
	});

	it("protects both profiles when the alternate profile already has a primary and fallback", async (t) => {
		const alternate = profile(b);
		const alternatePath = join(alternate, "auth.json");
		const beforeA = readFileSync(defaultAuthPath);
		const beforeB = readFileSync(alternatePath);
		const runtime = await veniceRuntime(t, alternate);

		await assert.rejects(runtime.login("venice", "api_key", interaction()), /add a fallback/);

		assert.deepEqual(readFileSync(alternatePath), beforeB);
		assert.deepEqual(readFileSync(defaultAuthPath), beforeA);
	});

	it("saves only the entered primary in an empty alternate profile", async (t) => {
		const alternate = profile();
		const beforeA = readFileSync(defaultAuthPath);
		const runtime = await veniceRuntime(t, alternate);

		await runtime.login("venice", "api_key", interaction("new-key"));

		assert.deepEqual(JSON.parse(readFileSync(join(alternate, "auth.json"), "utf8")), {
			venice: { type: "api_key", key: "new-key" },
		});
		assert.deepEqual(readFileSync(defaultAuthPath), beforeA);
	});

	it("protects an alternate profile even when its resolved key and fallbacks match the default", async (t) => {
		const alternate = profile({ ...a, key: "!echo a-1" });
		const alternatePath = join(alternate, "auth.json");
		const beforeB = readFileSync(alternatePath);
		const runtime = await veniceRuntime(t, alternate);
		assert.equal((await runtime.getAuth("venice"))?.auth.apiKey, a.key);

		await assert.rejects(runtime.login("venice", "api_key", interaction()), /add a fallback/);
		assert.deepEqual(readFileSync(alternatePath), beforeB);
	});

	for (const key of ["$PV_TEST_PRIMARY", "!echo dummy-primary"]) {
		it(`preserves a default profile with a ${key[0]} reference and numbered fallbacks`, async (t) => {
			const venice = { ...a, key, env: { ...a.env, PV_TEST_PRIMARY: "dummy-primary", VENICE_API_KEY_4: "a-4" } };
			const contents = `\uFEFF${JSON.stringify({ venice })}\n`;
			writeFileSync(defaultAuthPath, contents);
			const runtime = await veniceRuntime(t);

			await assert.rejects(runtime.login("venice", "api_key", interaction()), /add a fallback/);
			assert.equal(readFileSync(defaultAuthPath, "utf8"), contents);
		});
	}

	it("uses the injected store even when its agentDir points to a populated profile", async (t) => {
		const credentials = new InMemoryCredentialStore();
		const beforeA = readFileSync(defaultAuthPath);
		const runtime = await veniceRuntime(t, defaultDir, await ModelRuntime.create({ credentials, modelsPath: null }));

		await runtime.login("venice", "api_key", interaction("injected-primary"));
		assert.deepEqual(await credentials.read("venice"), { type: "api_key", key: "injected-primary" });

		await credentials.modify("venice", async () => b);
		await assert.rejects(runtime.login("venice", "api_key", interaction()), /add a fallback/);
		assert.deepEqual(await credentials.read("venice"), b);
		assert.deepEqual(readFileSync(defaultAuthPath), beforeA);
	});

	for (const key of ["!false", "$PV_TEST_UNSET_KEY"]) {
		it(`re-enters a stored ${key[0]} key that does not resolve and keeps its fallbacks`, async (t) => {
			const env = { ...a.env, VENICE_API_KEY_4: "a-4", PV_TEST_OTHER: "kept" };
			writeFileSync(defaultAuthPath, JSON.stringify({ venice: { type: "api_key", key, env } }));
			const runtime = await veniceRuntime(t);
			assert.equal(await runtime.getAuth("venice"), undefined);

			await runtime.login("venice", "api_key", interaction("new-key"));

			assert.deepEqual(JSON.parse(readFileSync(defaultAuthPath, "utf8")), {
				venice: { type: "api_key", key: "new-key", env },
			});
			assert.equal(runtime.getProviderAuthStatus("venice").configured, true);
			const auth = await runtime.getAuth("venice");
			assert.deepEqual(candidateKeys(auth?.auth.apiKey, auth?.env), ["new-key", "a-2", "a-4"]);
		});
	}

	it("re-enters a stored key that does not resolve and has no env as just the new key", async (t) => {
		writeFileSync(defaultAuthPath, JSON.stringify({ venice: { type: "api_key", key: "!false" } }));
		const runtime = await veniceRuntime(t);

		await runtime.login("venice", "api_key", interaction("new-key"));

		assert.deepEqual(JSON.parse(readFileSync(defaultAuthPath, "utf8")), {
			venice: { type: "api_key", key: "new-key" },
		});
	});

	it("keeps an environment-configured user's stored key that does not resolve", async (t) => {
		const contents = JSON.stringify({ venice: { type: "api_key", key: "!false", env: a.env } });
		writeFileSync(defaultAuthPath, contents);
		process.env.VENICE_API_KEY = "env-primary";
		const runtime = await veniceRuntime(t);
		// Without a stored key, Pi resolves VENICE_API_KEY, which a new stored key would override.
		assert.equal((await runtime.getAuth("venice"))?.auth.apiKey, "env-primary");

		await assert.rejects(runtime.login("venice", "api_key", interaction()), /export VENICE_API_KEY_2=<key>/);
		assert.equal(readFileSync(defaultAuthPath, "utf8"), contents);
	});

	it("checks for a working entry added while the prompt is open", async (t) => {
		const alternate = profile();
		const alternatePath = join(alternate, "auth.json");
		const runtime = await veniceRuntime(t, alternate);
		const contents = JSON.stringify({ venice: { type: "api_key", key: "!echo b-1", env: b.env } });
		await assert.rejects(runtime.login("venice", "api_key", {
			...interaction(),
			prompt: async () => {
				writeFileSync(alternatePath, contents);
				return "new-key";
			},
		}), /add a fallback/);
		assert.equal(readFileSync(alternatePath, "utf8"), contents);
	});

	it("keeps the fallbacks of an entry added while the prompt is open whose primary cannot resolve", async (t) => {
		const alternate = profile();
		const alternatePath = join(alternate, "auth.json");
		const runtime = await veniceRuntime(t, alternate);
		await runtime.login("venice", "api_key", {
			...interaction(),
			prompt: async () => {
				writeFileSync(alternatePath, JSON.stringify({ venice: { type: "api_key", key: "$PV_TEST_UNSET_KEY", env: b.env } }));
				return "new-key";
			},
		});
		assert.deepEqual(JSON.parse(readFileSync(alternatePath, "utf8")), {
			venice: { type: "api_key", key: "new-key", env: b.env },
		});
	});

	it("stops when a stored key that did not resolve is fixed while the prompt is open", async (t) => {
		writeFileSync(defaultAuthPath, JSON.stringify({ venice: { type: "api_key", key: "!false", env: a.env } }));
		const runtime = await veniceRuntime(t);
		const contents = JSON.stringify({ venice: { type: "api_key", key: "fixed-key", env: a.env } });
		await assert.rejects(runtime.login("venice", "api_key", {
			...interaction(),
			prompt: async () => {
				writeFileSync(defaultAuthPath, contents);
				return "new-key";
			},
		}), /add a fallback/);
		assert.equal(readFileSync(defaultAuthPath, "utf8"), contents);
	});

	it("stops when the active store fails during the login check", async (t) => {
		const alternate = profile();
		const alternatePath = join(alternate, "auth.json");
		const runtime = await veniceRuntime(t, alternate);
		await assert.rejects(runtime.login("venice", "api_key", {
			...interaction(),
			prompt: async () => {
				writeFileSync(alternatePath, "{ invalid");
				return "new-key";
			},
		}), /Credential store read failed/);
		assert.equal(readFileSync(alternatePath, "utf8"), "{ invalid");
	});

	it("stops when the session registry is unavailable", async () => {
		await assert.rejects(veniceApiKeyAuth(() => undefined).login!(interaction()), /could not verify/);
	});

	it("tells an environment-configured user which variable to export", async (t) => {
		const alternate = profile();
		process.env.VENICE_API_KEY = "env-primary";
		const runtime = await veniceRuntime(t, alternate);
		await assert.rejects(runtime.login("venice", "api_key", interaction()), /export VENICE_API_KEY_2=<key>/);
		assert.deepEqual(JSON.parse(readFileSync(join(alternate, "auth.json"), "utf8")), {});
	});
});

describe("nextFallbackIndex", () => {
	afterEach(() => {
		delete process.env.VENICE_API_KEY_2;
		delete process.env.VENICE_API_KEY_4;
	});

	it("skips indices already present in either environment", () => {
		process.env.VENICE_API_KEY_2 = "key-2";
		assert.equal(nextFallbackIndex({ VENICE_API_KEY_4: "key-4" }), 3);
	});
});
