/**
 * Integration tests: the provider's request pipeline (`veniceApi`) driven through pi's real
 * OpenAI-completions adapter, against a local HTTP server that speaks Venice's chat completions
 * wire format. The unit tests fake the adapter; these check that the Venice layers still line up
 * with how the adapter actually builds request bodies, parses streams, and reports errors.
 *
 * No request leaves the machine: every model's baseUrl points at the local server.
 */

import assert from "node:assert/strict";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { after, afterEach, before, beforeEach, describe, it } from "node:test";
import { normalizeContext } from "@earendil-works/pi-ai";
import type {
	AssistantMessage,
	AssistantMessageEventStream,
	Context,
	Model,
	SimpleStreamOptions,
	Usage,
} from "@earendil-works/pi-ai";
import { toPiModel, veniceApi } from "../extensions/venice/index.ts";
import { Keyring } from "../extensions/venice/keyring.ts";

// =============================================================================
// Mock Venice server
// =============================================================================

interface ReceivedRequest {
	key: string;
	headers: IncomingMessage["headers"];
	body: Record<string, unknown>;
}

type Respond = (request: ReceivedRequest, res: ServerResponse) => void;

let server: Server;
let baseUrl: string;
let received: ReceivedRequest[] = [];
let respond: Respond = (_request, res) => streamText(res, "ok");

function readBody(req: IncomingMessage): Promise<string> {
	return new Promise((resolve, reject) => {
		let data = "";
		req.setEncoding("utf8");
		req.on("data", (chunk: string) => {
			data += chunk;
		});
		req.on("end", () => resolve(data));
		req.on("error", reject);
	});
}

/** Write a server-sent event stream the way Venice does, ending with `[DONE]`. */
function sse(res: ServerResponse, chunks: unknown[]): void {
	res.writeHead(200, { "content-type": "text/event-stream", connection: "close" });
	for (const chunk of chunks) res.write(`data: ${JSON.stringify(chunk)}\n\n`);
	res.write("data: [DONE]\n\n");
	res.end();
}

const chunk = (delta: Record<string, unknown>, finishReason: string | null = null) => ({
	id: "chatcmpl-test",
	object: "chat.completion.chunk",
	created: 0,
	model: "test",
	choices: [{ index: 0, delta, finish_reason: finishReason }],
});

/** The final chunk: no choices, the usage, and (as Venice sends it) the billed cost at the top level. */
const usageChunk = (usage: Record<string, unknown>, cost?: { usd: number; diem: number }) => ({
	id: "chatcmpl-test",
	object: "chat.completion.chunk",
	created: 0,
	model: "test",
	choices: [],
	usage,
	...(cost ? { cost } : {}),
});

function streamText(
	res: ServerResponse,
	text: string,
	usage: Record<string, unknown> = { prompt_tokens: 10, completion_tokens: 2 },
	cost?: { usd: number; diem: number },
) {
	sse(res, [chunk({ role: "assistant", content: text }), chunk({}, "stop"), usageChunk(usage, cost)]);
}

/** Venice's documented body for a key without USD/DIEM balance. */
function insufficientBalance(res: ServerResponse): void {
	res.writeHead(402, { "content-type": "application/json", connection: "close" });
	res.end(
		JSON.stringify({
			code: "INSUFFICIENT_BALANCE",
			error: "Insufficient USD or Diem balance to complete request. Visit https://venice.ai/settings/api to add credits.",
		}),
	);
}

before(async () => {
	server = createServer(async (req, res) => {
		const raw = await readBody(req);
		const request: ReceivedRequest = {
			key: (req.headers.authorization ?? "").replace(/^Bearer\s+/i, ""),
			headers: req.headers,
			body: raw ? (JSON.parse(raw) as Record<string, unknown>) : {},
		};
		received.push(request);
		respond(request, res);
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

after(async () => {
	server.closeAllConnections();
	await new Promise<void>((resolve) => server.close(() => resolve()));
});

// Fallback keys and cache retention are read from the process environment as well; keep the
// developer's shell out of these tests.
const savedEnv: Record<string, string | undefined> = {};
const isolatedEnv = (name: string) => /^VENICE_API_KEY_\d+$/.test(name) || name === "PI_CACHE_RETENTION";

beforeEach(() => {
	received = [];
	respond = (_request, res) => streamText(res, "ok");
	for (const name of Object.keys(process.env).filter(isolatedEnv)) {
		savedEnv[name] = process.env[name];
		delete process.env[name];
	}
});

afterEach(() => {
	for (const [name, value] of Object.entries(savedEnv)) {
		if (value !== undefined) process.env[name] = value;
		delete savedEnv[name];
	}
});

// =============================================================================
// Models and requests
// =============================================================================

/** A catalog entry shaped like `GET /models?type=text`, mapped by the extension's own mapper. */
function catalogModel(
	id: string,
	capabilities: Record<string, unknown>,
	pricing: Record<string, { usd: number }> = { input: { usd: 1 }, output: { usd: 2 } },
): Model<"openai-completions"> {
	const model = toPiModel({
		id,
		type: "text",
		model_spec: {
			name: id,
			privacy: "anonymized",
			availableContextTokens: 200_000,
			maxCompletionTokens: 32_000,
			capabilities: { supportsFunctionCalling: true, ...capabilities },
			pricing,
		},
	});
	assert.ok(model, `${id} should be kept by the catalog filter`);
	return { ...model, baseUrl };
}

const claude = () =>
	catalogModel(
		"claude-opus-5-5",
		{ supportsReasoning: true, supportsReasoningEffort: true, reasoningEffortOptions: ["low", "medium", "high", "xhigh", "max"] },
		{ input: { usd: 4 }, output: { usd: 20 }, cache_input: { usd: 0.2 }, cache_write: { usd: 5 } },
	);

/** On the verified list: no effort control, but `disable_thinking` really stops reasoning. */
const grok = () => catalogModel("grok-4-20", { supportsReasoning: true, supportsReasoningEffort: false });

/** Lists a "none" effort, so pi turns thinking off with `reasoning_effort: "none"`. */
const kimi = () =>
	catalogModel("kimi-k3", { supportsReasoning: true, supportsReasoningEffort: true, reasoningEffortOptions: ["none", "low", "high", "max"] });

const tools = [
	{
		name: "read",
		description: "Read a file",
		parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
	},
];

const zeroUsage = (): Usage => ({
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});

/** One user turn, one tool call, and its result, as pi's agent loop sends after the first read. */
function toolLoopContext(model: Model<"openai-completions">, toolOutput: string): Context {
	return {
		systemPrompt: "You are a coding assistant.",
		tools,
		messages: [
			{ role: "user", content: "Read README.md", timestamp: 1 },
			{
				role: "assistant",
				content: [{ type: "toolCall", id: "call_1", name: "read", arguments: { path: "README.md" } }],
				api: model.api,
				provider: model.provider,
				model: model.id,
				usage: zeroUsage(),
				stopReason: "toolUse",
				timestamp: 2,
			},
			{
				role: "toolResult",
				toolCallId: "call_1",
				toolName: "read",
				content: [{ type: "text", text: toolOutput }],
				isError: false,
				timestamp: 3,
			},
		],
	};
}

const userContext = (text = "hi"): Context => ({
	systemPrompt: "You are a coding assistant.",
	messages: [{ role: "user", content: text, timestamp: 1 }],
});

async function run(
	api: ReturnType<typeof veniceApi>,
	model: Model<"openai-completions">,
	context: Context,
	options: SimpleStreamOptions,
): Promise<{ message: AssistantMessage; events: string[] }> {
	const stream: AssistantMessageEventStream = api.streamSimple(model, normalizeContext(context), options);
	const events: string[] = [];
	for await (const event of stream) events.push(event.type);
	return { message: await stream.result(), events };
}

const messagesSent = (index = 0) => received[index]!.body.messages as Record<string, unknown>[];

const near = (actual: number, expected: number) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);

// =============================================================================
// Tests
// =============================================================================

describe("request body through the real adapter", () => {
	it("adds the Venice defaults to the body pi builds", async () => {
		const { message } = await run(veniceApi(new Keyring()), claude(), userContext(), {
			apiKey: "key-1",
			sessionId: "session-1",
			reasoning: "high",
			maxTokens: 1000,
		});

		assert.equal(message.stopReason, "stop");
		assert.equal(received.length, 1);
		const { key, body } = received[0]!;
		assert.equal(key, "key-1");
		assert.equal(body.model, "claude-opus-5-5");
		assert.equal(body.stream, true);
		assert.deepEqual(body.stream_options, { include_usage: true });
		assert.deepEqual(body.venice_parameters, { include_venice_system_prompt: false });
		assert.equal(body.prompt_cache_key, "session-1");
		assert.equal(body.reasoning_effort, "high");
		assert.equal(body.max_completion_tokens, 1000);
		assert.equal("max_tokens" in body, false);
		assert.equal("store" in body, false);
		// Reasoning models get the developer role.
		assert.equal(messagesSent()[0]!.role, "developer");
	});

	// pi sends thinking level off as an absent `reasoning`.
	it("turns thinking off with disable_thinking on a verified model", async () => {
		await run(veniceApi(new Keyring()), grok(), userContext(), { apiKey: "key-1" });

		const { body } = received[0]!;
		assert.deepEqual(body.venice_parameters, { include_venice_system_prompt: false, disable_thinking: true });
		assert.equal("reasoning_effort" in body, false);
	});

	it("turns thinking off with reasoning_effort none where Venice lists it", async () => {
		await run(veniceApi(new Keyring()), kimi(), userContext(), { apiKey: "key-1" });

		const { body } = received[0]!;
		assert.equal(body.reasoning_effort, "none");
		assert.deepEqual(body.venice_parameters, { include_venice_system_prompt: false });
	});

	it("merges user venice_parameters layers with the defaults", async () => {
		const model = {
			...kimi(),
			samplingParams: { venice_parameters: { enable_web_search: "auto" } },
			samplingParamsByThinkingLevel: { high: { venice_parameters: { enable_web_citations: true } } },
		};
		await run(veniceApi(new Keyring()), model, userContext(), { apiKey: "key-1", reasoning: "high" });

		assert.deepEqual(received[0]!.body.venice_parameters, {
			include_venice_system_prompt: false,
			enable_web_search: "auto",
			enable_web_citations: true,
		});
	});

	it("places Claude cache breakpoints in the forms Venice honors", async () => {
		const model = claude();
		await run(veniceApi(new Keyring()), model, toolLoopContext(model, "# README"), {
			apiKey: "key-1",
			sessionId: "session-1",
			reasoning: "high",
		});

		const [instruction, , , toolResult] = messagesSent();
		// The instruction message: on its text part (Venice adds none of its own for `developer`).
		assert.equal(instruction!.role, "developer");
		const parts = instruction!.content as Record<string, unknown>[];
		assert.deepEqual(parts.at(-1)!.cache_control, { type: "ephemeral" });
		// The last turn is a tool result: on the message itself, the only form Venice reads there.
		assert.equal(toolResult!.role, "tool");
		assert.deepEqual(toolResult!.cache_control, { type: "ephemeral" });
		assert.equal(typeof toolResult!.content, "string");
	});

	it("leaves non-Claude requests and caching-off requests without breakpoints", async () => {
		const api = veniceApi(new Keyring());
		const other = kimi();
		await run(api, other, toolLoopContext(other, "# README"), { apiKey: "key-1", sessionId: "s" });
		const model = claude();
		await run(api, model, toolLoopContext(model, "# README"), { apiKey: "key-1", sessionId: "s", cacheRetention: "none" });

		for (const index of [0, 1]) {
			assert.equal(JSON.stringify(messagesSent(index)).includes("cache_control"), false);
		}
		assert.equal("prompt_cache_key" in received[1]!.body, false);
	});

	it("drops NUL bytes from a tool result before sending", async () => {
		const model = kimi();
		await run(veniceApi(new Keyring()), model, toolLoopContext(model, "ELF\u0000\u0000binary\u0000tail"), { apiKey: "key-1" });

		const toolResult = messagesSent().find((message) => message.role === "tool")!;
		assert.equal(toolResult.content, "ELFbinarytail");
	});
});

describe("TEE models through the real adapter", () => {
	/** E2EE-capable, like every TEE model in Venice's catalog; this provider runs them TEE-only. */
	const tee = (id: string, capabilities: Record<string, unknown> = {}) =>
		catalogModel(id, { supportsE2EE: true, supportsTeeAttestation: true, supportsReasoning: true, ...capabilities });

	it("sends plaintext requests without E2EE headers", async () => {
		const model = tee("e2ee-deepseek-v4-flash");
		const { message } = await run(veniceApi(new Keyring()), model, toolLoopContext(model, "# README"), {
			apiKey: "key-1",
			sessionId: "session-1",
		});

		assert.equal(message.stopReason, "stop");
		const { headers, body } = received[0]!;
		// Venice switches to E2EE only when these headers come with client-encrypted messages.
		assert.deepEqual(
			Object.keys(headers).filter((name) => name.startsWith("x-venice-tee-")),
			[],
		);
		assert.equal(body.model, "e2ee-deepseek-v4-flash");
		assert.ok(Array.isArray(body.tools) && body.tools.length === 1);
		const user = messagesSent().find((entry) => entry.role === "user")!;
		assert.equal(user.content, "Read README.md");
		assert.deepEqual(body.venice_parameters, { include_venice_system_prompt: false });
	});

	it("never sends the none effort that Venice rejects in TEE-only mode", async () => {
		const model = tee("e2ee-glm-5-3-p", { supportsReasoningEffort: true, reasoningEffortOptions: ["none", "low", "medium", "high"] });
		const api = veniceApi(new Keyring());
		// No level: the model's default effort, rather than reasoning_effort "none".
		await run(api, model, userContext(), { apiKey: "key-1" });
		// pi's session clamps a stored "off" to the lowest offered level before it sends a request.
		await run(api, model, userContext(), { apiKey: "key-1", reasoning: "low" });

		assert.equal("reasoning_effort" in received[0]!.body, false);
		assert.equal(received[1]!.body.reasoning_effort, "low");
		for (const { body } of received) {
			assert.deepEqual(body.venice_parameters, { include_venice_system_prompt: false });
		}
	});
});

describe("usage and cost through the real adapter", () => {
	it("moves Venice's cache_creation_input_tokens onto cacheWrite and prices them", async () => {
		respond = (_request, res) =>
			streamText(res, "ok", {
				prompt_tokens: 15637,
				completion_tokens: 4,
				total_tokens: 15641,
				prompt_tokens_details: { cached_tokens: 0, cache_creation_input_tokens: 15621 },
				cache_creation_input_tokens: 15621,
			});
		const { message } = await run(veniceApi(new Keyring()), claude(), userContext(), { apiKey: "key-1" });

		assert.equal(message.usage.input, 16);
		assert.equal(message.usage.cacheWrite, 15621);
		assert.equal(message.usage.cacheRead, 0);
		assert.equal(message.usage.totalTokens, 15641);
		near(message.usage.cost.cacheWrite, (15621 * 5) / 1e6);
		near(message.usage.cost.total, (16 * 4 + 4 * 20 + 15621 * 5) / 1e6);
		// No `cost` in the chunk: the total is the catalog estimate.
		assert.notEqual(message.usage.cost.totalFromProvider, true);
	});

	it("shows the cost Venice billed, including charges outside the token prices", async () => {
		// Shaped after a live web search turn: Venice billed the tokens plus about $0.01 for the search.
		const tokens = (3369 * 4 + 48 * 20) / 1e6;
		respond = (_request, res) =>
			streamText(res, "ok", { prompt_tokens: 3369, completion_tokens: 48, prompt_tokens_details: { cached_tokens: 0 } }, {
				usd: 0,
				diem: tokens + 0.01,
			});
		const { message } = await run(veniceApi(new Keyring()), claude(), userContext(), { apiKey: "key-1" });

		near(message.usage.cost.total, tokens + 0.01);
		assert.equal(message.usage.cost.totalFromProvider, true);
		// The per-category amounts stay catalog estimates.
		near(message.usage.cost.input + message.usage.cost.output, tokens);
	});

	it("combines the cache-write correction with the billed total", async () => {
		const billed = (16 * 4 + 4 * 20 + 15621 * 5) / 1e6;
		respond = (_request, res) =>
			streamText(
				res,
				"ok",
				{ prompt_tokens: 15637, completion_tokens: 4, prompt_tokens_details: { cached_tokens: 0, cache_creation_input_tokens: 15621 } },
				{ usd: 0.00005, diem: billed - 0.00005 },
			);
		const { message } = await run(veniceApi(new Keyring()), claude(), userContext(), { apiKey: "key-1" });

		assert.equal(message.usage.input, 16);
		assert.equal(message.usage.cacheWrite, 15621);
		near(message.usage.cost.total, billed);
		assert.equal(message.usage.cost.totalFromProvider, true);
	});

	it("leaves cache reads to the adapter", async () => {
		respond = (_request, res) =>
			streamText(res, "ok", { prompt_tokens: 15656, completion_tokens: 7, prompt_tokens_details: { cached_tokens: 15621 } });
		const { message } = await run(veniceApi(new Keyring()), claude(), userContext(), { apiKey: "key-1" });

		assert.equal(message.usage.input, 35);
		assert.equal(message.usage.cacheRead, 15621);
		assert.equal(message.usage.cacheWrite, 0);
		near(message.usage.cost.total, (35 * 4 + 7 * 20 + 15621 * 0.2) / 1e6);
	});
});

describe("key rotation through the real adapter", () => {
	const options = (extra: Partial<SimpleStreamOptions> = {}): SimpleStreamOptions => ({
		apiKey: "key-1",
		env: { VENICE_API_KEY_2: "key-2" },
		...extra,
	});

	it("retries a 402 on the fallback key and keeps using it", async () => {
		respond = (request, res) => (request.key === "key-1" ? insufficientBalance(res) : streamText(res, `ok from ${request.key}`));
		const api = veniceApi(new Keyring());

		const first = await run(api, kimi(), userContext(), options());
		const second = await run(api, kimi(), userContext(), options());

		assert.deepEqual(
			received.map((request) => request.key),
			["key-1", "key-2", "key-2"],
		);
		assert.equal(first.message.stopReason, "stop");
		assert.equal(first.message.content.find((block) => block.type === "text")?.text, "ok from key-2");
		// The failed attempt is invisible: one start, one done.
		assert.equal(first.events.filter((type) => type === "start").length, 1);
		assert.equal(first.events.at(-1), "done");
		assert.equal(second.message.stopReason, "stop");
	});

	it("sends the same Venice defaults on the retried key", async () => {
		respond = (request, res) => (request.key === "key-1" ? insufficientBalance(res) : streamText(res, "ok"));
		await run(veniceApi(new Keyring()), grok(), userContext(), options({ sessionId: "session-1" }));

		assert.equal(received.length, 2);
		assert.equal((received[0]!.body.venice_parameters as Record<string, unknown>).disable_thinking, true);
		assert.deepEqual(received[1]!.body.venice_parameters, received[0]!.body.venice_parameters);
		assert.equal(received[1]!.body.prompt_cache_key, "session-1");
	});

	it("passes a 402 through when no fallback key is configured", async () => {
		respond = (_request, res) => insufficientBalance(res);
		const { message } = await run(veniceApi(new Keyring()), kimi(), userContext(), { apiKey: "key-1" });

		assert.equal(received.length, 1);
		assert.equal(message.stopReason, "error");
		assert.match(message.errorMessage ?? "", /^402\b/);
	});

	it("does not rotate on other client errors", async () => {
		respond = (_request, res) => {
			res.writeHead(400, { "content-type": "application/json", connection: "close" });
			res.end(JSON.stringify({ error: "Invalid request parameters" }));
		};
		const { message } = await run(veniceApi(new Keyring()), kimi(), userContext(), options());

		assert.deepEqual(
			received.map((request) => request.key),
			["key-1"],
		);
		assert.equal(message.stopReason, "error");
		assert.match(message.errorMessage ?? "", /^400\b/);
	});

	it("passes a balance failure after streaming started through, then uses the next key", async () => {
		respond = (request, res) => {
			if (request.key !== "key-1") return streamText(res, "ok");
			sse(res, [
				chunk({ role: "assistant", content: "Hel" }),
				{ error: { message: "402 Insufficient USD or Diem balance to complete request.", code: "INSUFFICIENT_BALANCE" } },
			]);
		};
		const api = veniceApi(new Keyring());

		const first = await run(api, kimi(), userContext(), options());
		const second = await run(api, kimi(), userContext(), options());

		assert.equal(first.message.stopReason, "error");
		assert.equal(first.message.content.find((block) => block.type === "text")?.text, "Hel");
		assert.equal(second.message.stopReason, "stop");
		assert.deepEqual(
			received.map((request) => request.key),
			["key-1", "key-2"],
		);
	});
});
