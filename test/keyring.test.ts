import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import type {
	AssistantMessage,
	AssistantMessageEventStream,
	Model,
	ProviderStreams,
	SimpleStreamOptions,
	StreamOptions,
} from "@earendil-works/pi-ai";
import { candidateKeys, withKeyRotation } from "../extensions/venice/keyring.ts";

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
