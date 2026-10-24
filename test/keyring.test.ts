import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, it } from "node:test";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import type {
	ApiKeyCredential,
	AssistantMessage,
	AssistantMessageEventStream,
	Model,
	ProviderAuthInteraction,
	ProviderStreams,
	SimpleStreamOptions,
	StreamOptions,
} from "@earendil-works/pi-ai";
import { candidateKeys, nextFallbackIndex, veniceApiKeyAuth, withKeyRotation } from "../extensions/venice/keyring.ts";

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

const outcome = (kind: "done" | "started-error" | "error", message: string): AssistantMessageEventStream => {
	const stream = createAssistantMessageEventStream();
	const result = assistant(message);
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
	return stream;
};

/** Fake provider that records the key each attempt used. */
function fakeApi(
	respond: (apiKey: string | undefined) => AssistantMessageEventStream,
): ProviderStreams & { calls: (string | undefined)[] } {
	const calls: (string | undefined)[] = [];
	const produce = (options: StreamOptions | undefined) => {
		calls.push(options?.apiKey);
		return respond(options?.apiKey);
	};
	return {
		calls,
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

	it("retries on the fallback key when Venice reports insufficient balance", async () => {
		const api = fakeApi((key) =>
			key === "key-1" ? outcome("error", "402: {\"code\":\"INSUFFICIENT_BALANCE\"}") : outcome("done", "ok"),
		);
		const rotated = withKeyRotation(api);
		const result = await drain(call(rotated, options("key-1", fallback)));

		assert.deepEqual(api.calls, ["key-1", "key-2"]);
		assert.equal(result.stopReason, "stop");
		assert.equal(result.content[0]?.type === "text" && result.content[0].text, "ok");
	});

	it("keeps using the fallback key for later requests", async () => {
		const api = fakeApi((key) => (key === "key-1" ? outcome("error", "402 Payment Required") : outcome("done", "ok")));
		const rotated = withKeyRotation(api);

		await drain(call(rotated, options("key-1", fallback)));
		await drain(call(rotated, options("key-1", fallback)));

		assert.deepEqual(api.calls, ["key-1", "key-2", "key-2"]);
	});

	it("does not retry a balance failure after the response started", async () => {
		const api = fakeApi(() => outcome("started-error", "402: insufficient balance"));
		const rotated = withKeyRotation(api);
		const result = await drain(call(rotated, options("key-1", fallback)));

		assert.deepEqual(api.calls, ["key-1"]);
		assert.equal(result.stopReason, "error");
	});

	it("remembers the next key after a late balance failure without replaying", async () => {
		const api = fakeApi((key) =>
			key === "key-1" ? outcome("started-error", "402: insufficient balance") : outcome("done", "ok"),
		);
		const rotated = withKeyRotation(api);

		await drain(call(rotated, options("key-1", fallback)));
		const result = await drain(call(rotated, options("key-1", fallback)));

		assert.deepEqual(api.calls, ["key-1", "key-2"]);
		assert.equal(result.stopReason, "stop");
	});

	it("does not retry unrelated errors", async () => {
		const api = fakeApi(() => outcome("error", "500: internal server error"));
		const rotated = withKeyRotation(api);
		const result = await drain(call(rotated, options("key-1", fallback)));

		assert.deepEqual(api.calls, ["key-1"]);
		assert.equal(result.stopReason, "error");
	});

	it("passes the balance error through when no fallback is configured", async () => {
		const api = fakeApi(() => outcome("error", "402: insufficient balance"));
		const rotated = withKeyRotation(api);
		const result = await drain(call(rotated, options("key-1")));

		assert.deepEqual(api.calls, ["key-1"]);
		assert.match(result.errorMessage ?? "", /402/);
	});

	it("retries the primary key on a later request when every key failed", async () => {
		const api = fakeApi(() => outcome("error", "402: insufficient balance"));
		const rotated = withKeyRotation(api);

		await drain(call(rotated, options("key-1", fallback)));
		await drain(call(rotated, options("key-1", fallback)));

		assert.deepEqual(api.calls, ["key-1", "key-2", "key-1", "key-2"]);
	});

	it("rotates streamSimple requests too", async () => {
		const api = fakeApi((key) => (key === "key-1" ? outcome("error", "402") : outcome("done", "ok")));
		const rotated = withKeyRotation(api);
		const result = await drain(simpleCall(rotated, { apiKey: "key-1", env: fallback }));

		assert.deepEqual(api.calls, ["key-1", "key-2"]);
		assert.equal(result.stopReason, "stop");
	});
});

describe("veniceApiKeyAuth login", () => {
	const savedAgentDir = process.env.PI_CODING_AGENT_DIR;
	const savedPrimary = process.env.VENICE_API_KEY;
	const tempDirs: string[] = [];

	function useAuthFile(credential?: unknown): void {
		const dir = mkdtempSync(join(tmpdir(), "pi-venice-auth-"));
		tempDirs.push(dir);
		process.env.PI_CODING_AGENT_DIR = dir;
		if (credential !== undefined) writeFileSync(join(dir, "auth.json"), JSON.stringify(credential));
	}

	function interaction(answers: string[]): ProviderAuthInteraction & { prompts: string[]; notices: string[] } {
		const prompts: string[] = [];
		const notices: string[] = [];
		return {
			signal: new AbortController().signal,
			prompts,
			notices,
			prompt: async (prompt) => {
				prompts.push(prompt.message);
				const answer = answers.shift();
				if (answer === undefined) throw new Error("No answer prepared for prompt");
				return answer;
			},
			notify: (event) => {
				if (event.type === "info") notices.push(event.message);
			},
		};
	}

	afterEach(() => {
		for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
		if (savedAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = savedAgentDir;
		if (savedPrimary === undefined) delete process.env.VENICE_API_KEY;
		else process.env.VENICE_API_KEY = savedPrimary;
	});

	it("stores the primary key on the first login", async () => {
		useAuthFile();
		delete process.env.VENICE_API_KEY;
		const flow = interaction(["key-1"]);
		const credential = await veniceApiKeyAuth().login!(flow);

		assert.deepEqual(credential, { type: "api_key", key: "key-1" });
		assert.deepEqual(flow.prompts, ["Enter Venice API key"]);
	});

	it("stores the second login as VENICE_API_KEY_2 and keeps the primary", async () => {
		useAuthFile({ venice: { type: "api_key", key: "key-1", env: { EXISTING: "1" } } });
		delete process.env.VENICE_API_KEY;
		const flow = interaction(["key-2"]);
		const credential = await veniceApiKeyAuth().login!(flow);

		assert.deepEqual(credential, {
			type: "api_key",
			key: "key-1",
			env: { EXISTING: "1", VENICE_API_KEY_2: "key-2" },
		});
		assert.match(flow.notices[0] ?? "", /VENICE_API_KEY_2/);
	});

	it("appends the next free numbered key", async () => {
		useAuthFile({ venice: { type: "api_key", key: "key-1", env: { VENICE_API_KEY_2: "key-2" } } });
		delete process.env.VENICE_API_KEY;
		const flow = interaction(["key-3"]);
		const credential = (await veniceApiKeyAuth().login!(flow)) as ApiKeyCredential;

		assert.deepEqual(credential.env, { VENICE_API_KEY_2: "key-2", VENICE_API_KEY_3: "key-3" });
	});

	it("tells an environment-configured user which variable to export", async () => {
		useAuthFile();
		process.env.VENICE_API_KEY = "key-1";
		const flow = interaction([]);

		await assert.rejects(veniceApiKeyAuth().login!(flow), /export VENICE_API_KEY_2=<key>/);
		assert.deepEqual(flow.prompts, []);
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
