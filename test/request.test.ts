import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import { calculateCost, createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import type {
	AssistantMessage,
	AssistantMessageEventStream,
	ProviderStreams,
	SimpleStreamOptions,
	StreamOptions,
	Usage,
} from "@earendil-works/pi-ai";
import { applyVeniceDefaults, imagesToOmit, withVeniceRequestDefaults } from "../extensions/venice/request.ts";

const body = (extra: Record<string, unknown> = {}) => ({ model: "kimi-k3", messages: [], ...extra });

const zeroUsage = (): Usage => ({
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});

const assistant = (usage: Usage = zeroUsage()): AssistantMessage => ({
	role: "assistant",
	content: [],
	api: "fake",
	provider: "fake",
	model: "fake",
	usage,
	stopReason: "stop",
	timestamp: Date.now(),
});

/** Consume a provider stream the way pi does, returning the final message. */
async function drain(stream: AssistantMessageEventStream): Promise<AssistantMessage | undefined> {
	for await (const _event of stream) {
		// drain
	}
	return await stream.result();
}

const near = (actual: number, expected: number) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);

describe("applyVeniceDefaults", () => {
	const savedRetention = process.env.PI_CACHE_RETENTION;
	afterEach(() => {
		if (savedRetention === undefined) delete process.env.PI_CACHE_RETENTION;
		else process.env.PI_CACHE_RETENTION = savedRetention;
	});

	it("disables the Venice system prompt and adds the session cache key", () => {
		assert.deepEqual(applyVeniceDefaults(body(), { sessionId: "session-1" }), {
			...body(),
			venice_parameters: { include_venice_system_prompt: false },
			prompt_cache_key: "session-1",
		});
	});

	it("keeps user venice_parameters next to the default", () => {
		const result = applyVeniceDefaults(body({ venice_parameters: { enable_web_search: "on" } }), {});
		assert.deepEqual((result as Record<string, unknown>).venice_parameters, {
			include_venice_system_prompt: false,
			enable_web_search: "on",
		});
	});

	it("respects an explicit include_venice_system_prompt", () => {
		const result = applyVeniceDefaults(body({ venice_parameters: { include_venice_system_prompt: true } }), {});
		assert.deepEqual((result as Record<string, unknown>).venice_parameters, { include_venice_system_prompt: true });
	});

	it("leaves a non-object venice_parameters for Venice to reject", () => {
		const result = applyVeniceDefaults(body({ venice_parameters: "on" }), {});
		assert.equal((result as Record<string, unknown>).venice_parameters, "on");
	});

	it("keeps an existing prompt_cache_key", () => {
		const result = applyVeniceDefaults(body({ prompt_cache_key: "mine" }), { sessionId: "session-1" });
		assert.equal((result as Record<string, unknown>).prompt_cache_key, "mine");
	});

	it("omits the cache key without a session or with caching off", () => {
		delete process.env.PI_CACHE_RETENTION;
		const noSession = applyVeniceDefaults(body(), {}) as Record<string, unknown>;
		const optionOff = applyVeniceDefaults(body(), { sessionId: "s", cacheRetention: "none" }) as Record<string, unknown>;
		assert.equal("prompt_cache_key" in noSession, false);
		assert.equal("prompt_cache_key" in optionOff, false);
	});

	it("clamps the cache key to 64 characters", () => {
		const result = applyVeniceDefaults(body(), { sessionId: "x".repeat(80) }) as Record<string, unknown>;
		assert.equal(result.prompt_cache_key, "x".repeat(64));
	});

	it("does not mutate the input or touch non-object payloads", () => {
		const input = body({ venice_parameters: { enable_web_search: "on" } });
		const snapshot = structuredClone(input);
		applyVeniceDefaults(input, { sessionId: "s" });
		assert.deepEqual(input, snapshot);
		assert.equal(applyVeniceDefaults(null, { sessionId: "s" }), null);
	});
});

describe("NUL bytes in messages", () => {
	it("drops NUL bytes from message text, content parts and tool arguments", () => {
		const input = body({
			messages: [
				{ role: "tool", tool_call_id: "c1", content: "\u007fELF\u0002\u0001\u0001\u0000\u0000" },
				{
					role: "assistant",
					content: [{ type: "text", text: "a\u0000b" }],
					tool_calls: [{ id: "c2", type: "function", function: { name: "read", arguments: '{"p":"x\u0000y"}' } }],
				},
			],
		});
		const snapshot = structuredClone(input);

		const clean = applyVeniceDefaults(input, {}) as { messages: unknown };
		assert.deepEqual(clean.messages, [
			{ role: "tool", tool_call_id: "c1", content: "\u007fELF\u0002\u0001\u0001" },
			{
				role: "assistant",
				content: [{ type: "text", text: "ab" }],
				tool_calls: [{ id: "c2", type: "function", function: { name: "read", arguments: '{"p":"xy"}' } }],
			},
		]);
		assert.ok(!JSON.stringify(clean).includes("\\u0000"));
		assert.deepEqual(input, snapshot);
	});

	it("keeps other control characters and leaves clean messages by reference", () => {
		const messages = [{ role: "user", content: "a\u0001b\u001bc\ufffdd" }];
		const result = applyVeniceDefaults(body({ messages }), {}) as { messages: unknown };
		assert.equal(result.messages, messages);
	});
});

describe("image limit", () => {
	const image = (n: number) => ({ type: "image_url", image_url: { url: `data:image/png;base64,${n}` } });
	const omitted = (max: number) => ({ type: "text", text: `(image omitted: Venice accepts at most ${max} images per request)` });
	const screenshots = (count: number) =>
		Array.from({ length: count }, (_, i) => ({ role: "user", content: [{ type: "text", text: `screenshot ${i}` }, image(i)] }));
	const messagesOf = (payload: unknown) => (payload as { messages: { content: unknown[] }[] }).messages;
	const imagesIn = (messages: { content: unknown[] }[]) =>
		messages.flatMap((message) =>
			Array.isArray(message.content) ? message.content.filter((part) => (part as { type?: string }).type === "image_url") : [],
		);

	it("omits the oldest images in a block of 32 once over 64", () => {
		const messages = screenshots(65);
		const snapshot = structuredClone(messages);

		const result = messagesOf(applyVeniceDefaults(body({ messages }), {}));

		assert.equal(imagesIn(result).length, 33);
		assert.deepEqual(result[0]!.content, [{ type: "text", text: "screenshot 0" }, omitted(64)]);
		assert.deepEqual(result[31]!.content, [{ type: "text", text: "screenshot 31" }, omitted(64)]);
		assert.deepEqual(result[32]!.content, [{ type: "text", text: "screenshot 32" }, image(32)]);
		assert.deepEqual(result.at(-1)!.content, [{ type: "text", text: "screenshot 64" }, image(64)]);
		assert.deepEqual(messages, snapshot);
	});

	it("keeps the same images omitted until the next block is needed", () => {
		assert.equal(imagesToOmit(64, 64), 0);
		assert.equal(imagesToOmit(65, 64), 32);
		assert.equal(imagesToOmit(96, 64), 32);
		assert.equal(imagesToOmit(97, 64), 64);

		// The prefix up to the newest image of the shorter request is identical, so it stays cacheable.
		const shorter = messagesOf(applyVeniceDefaults(body({ messages: screenshots(65) }), {}));
		const longer = messagesOf(applyVeniceDefaults(body({ messages: screenshots(96) }), {}));
		assert.deepEqual(longer.slice(0, shorter.length), shorter);
		assert.equal(imagesIn(longer).length, 64);
	});

	it("uses the limit from the request context, with blocks no larger than the limit", () => {
		const messages = Array.from({ length: 3 }, (_, i) => ({ role: "user", content: [image(i)] }));

		const result = messagesOf(applyVeniceDefaults(body({ messages }), { maxImagesPerRequest: 2 }));

		assert.deepEqual(
			result.map((message) => message.content),
			[[omitted(2)], [omitted(2)], [image(2)]],
		);
	});

	it("leaves a request within the limit by reference", () => {
		const messages = Array.from({ length: 64 }, (_, i) => ({ role: "user", content: [image(i)] }));
		const result = messagesOf(applyVeniceDefaults(body({ messages }), {}));
		assert.equal(result, messages);
	});
});

describe("Claude cache breakpoints", () => {
	const cache = { type: "ephemeral" };
	const image = { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } };
	const toolCall = (id: string) => ({
		role: "assistant",
		content: null,
		tool_calls: [{ id, type: "function", function: { name: "read", arguments: "{}" } }],
	});
	const messagesOf = (payload: unknown) => (payload as { messages: Record<string, unknown>[] }).messages;
	const withBreakpoints = (messages: unknown[], extra: Record<string, unknown> = {}) =>
		messagesOf(applyVeniceDefaults(body({ messages }), { cacheBreakpoints: true, ...extra }));

	it("marks the developer message on a text part and the last tool result on the message", () => {
		const input = [
			{ role: "developer", content: "You are pi." },
			{ role: "user", content: "Read a file." },
			toolCall("c1"),
			{ role: "tool", tool_call_id: "c1", content: "first" },
			toolCall("c2"),
			{ role: "tool", tool_call_id: "c2", content: "second" },
		];
		const snapshot = structuredClone(input);
		const messages = withBreakpoints(input);

		assert.deepEqual(messages[0], {
			role: "developer",
			content: [{ type: "text", text: "You are pi.", cache_control: cache }],
		});
		assert.deepEqual(messages[5], { role: "tool", tool_call_id: "c2", content: "second", cache_control: cache });
		// Only the last turn gets a breakpoint; Venice places its own on the turn before.
		assert.deepEqual(messages.slice(1, 5), snapshot.slice(1, 5));
		assert.deepEqual(input, snapshot);
	});

	it("marks the last user turn on its last text part, never on an image", () => {
		const messages = withBreakpoints([
			{ role: "system", content: [{ type: "text", text: "You are pi." }] },
			{ role: "user", content: "Earlier turn." },
			{ role: "assistant", content: "Done." },
			{ role: "user", content: [{ type: "text", text: "Attached image(s) from tool result:" }, image] },
		]);

		assert.deepEqual(messages[0]?.content, [{ type: "text", text: "You are pi.", cache_control: cache }]);
		assert.equal(messages[1]?.content, "Earlier turn.");
		assert.deepEqual(messages[3], {
			role: "user",
			content: [{ type: "text", text: "Attached image(s) from tool result:", cache_control: cache }, image],
		});
	});

	it("falls back to the latest turn that can carry a breakpoint", () => {
		const messages = withBreakpoints([
			{ role: "developer", content: "You are pi." },
			{ role: "user", content: "Look at this." },
			toolCall("c1"),
			{ role: "tool", tool_call_id: "c1", content: "(see attached image)" },
			{ role: "user", content: [image] },
			{ role: "assistant", content: "Partial answer" },
		]);

		assert.deepEqual(messages[3]?.cache_control, cache);
		assert.deepEqual(messages[4], { role: "user", content: [image] });
		assert.equal(messages[5]?.content, "Partial answer");
	});

	it("keeps existing breakpoints", () => {
		const longCache = { type: "ephemeral", ttl: "1h" };
		const input = [
			{ role: "developer", content: [{ type: "text", text: "You are pi.", cache_control: longCache }] },
			{ role: "user", content: "Read a file." },
			toolCall("c1"),
			{ role: "tool", tool_call_id: "c1", content: "first", cache_control: longCache },
		];
		assert.deepEqual(withBreakpoints(input), input);
	});

	it("adds no breakpoints when prompt caching is off or the model does not need them", () => {
		const input = [
			{ role: "developer", content: "You are pi." },
			{ role: "user", content: "Hello" },
		];
		assert.deepEqual(withBreakpoints(input, { cacheRetention: "none" }), input);
		assert.deepEqual(messagesOf(applyVeniceDefaults(body({ messages: input }), {})), input);
	});

	it("is applied to Claude models only, before the caller's onPayload", async () => {
		const prompt = [
			{ role: "developer", content: "You are pi." },
			{ role: "user", content: "Hello" },
		];
		const seen: unknown[] = [];
		const api: ProviderStreams = {
			stream: () => {
				throw new Error("unused");
			},
			streamSimple: (model, _context, options) => {
				const stream = createAssistantMessageEventStream();
				void (async () => {
					await options?.onPayload?.(body({ messages: prompt }), model);
					stream.push({ type: "done", reason: "stop", message: assistant() });
					stream.end();
				})();
				return stream;
			},
		};
		const wrapped = withVeniceRequestDefaults(api, (_model, level) => level);
		const onPayload = (payload: unknown) => {
			seen.push(messagesOf(payload));
			return undefined;
		};

		await drain(wrapped.streamSimple({ id: "claude-opus-5-5" } as never, {} as never, { onPayload }));
		await drain(wrapped.streamSimple({ id: "kimi-k3" } as never, {} as never, { onPayload }));

		assert.deepEqual(seen[0], [
			{ role: "developer", content: [{ type: "text", text: "You are pi.", cache_control: cache }] },
			{ role: "user", content: [{ type: "text", text: "Hello", cache_control: cache }] },
		]);
		assert.deepEqual(seen[1], prompt);
	});
});

describe("withVeniceRequestDefaults", () => {
	type Level = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
	type FakeModel = {
		id?: string;
		reasoning?: boolean;
		thinkingLevelMap?: { off?: string | null };
		samplingParams?: Record<string, unknown>;
		samplingParamsByThinkingLevel?: Partial<Record<Level, Record<string, unknown>>>;
	};

	const identityClamp = (_model: unknown, level: Level) => level;

	/**
	 * Fake completions API: builds the body with pi's shallow merge (model, thinking level, request;
	 * later wins per top-level key), then sends it through options.onPayload.
	 */
	function fakeApi(sent: unknown[], clamp: (model: never, level: Level) => Level = identityClamp): ProviderStreams {
		const run = async (model: FakeModel, level: Level, options?: StreamOptions) => {
			const params = {
				...model.samplingParams,
				...model.samplingParamsByThinkingLevel?.[clamp(model as never, level)],
				...options?.samplingParams,
			};
			const built = { ...body(), ...params };
			const replaced = await options?.onPayload?.(built, model as never);
			sent.push(replaced === undefined ? built : replaced);
		};
		const streaming = (run: Promise<void>): AssistantMessageEventStream => {
			const stream = createAssistantMessageEventStream();
			void run.then(() => {
				stream.push({ type: "done", reason: "stop", message: assistant() });
				stream.end();
			});
			return stream;
		};
		return {
			stream: (model, _context, options) =>
				streaming(run(model as FakeModel, (options as { reasoningEffort?: Level })?.reasoningEffort ?? "off", options)),
			streamSimple: (model, _context, options) =>
				streaming(run(model as FakeModel, options?.reasoning ?? "off", options)),
		};
	}

	// Real models always carry an id; default to a model without explicit cache breakpoints.
	const streamCall = (api: ProviderStreams, model: FakeModel, options?: StreamOptions) =>
		drain(api.stream({ id: "kimi-k3", ...model } as never, {} as never, options));
	const simpleCall = (api: ProviderStreams, model: FakeModel, options?: SimpleStreamOptions) =>
		drain(api.streamSimple({ id: "kimi-k3", ...model } as never, {} as never, options));

	const veniceParametersSent = (sent: unknown[]) => (sent[0] as Record<string, unknown>).venice_parameters;

	it("applies defaults before the caller's onPayload, which can still replace the body", async () => {
		const sent: unknown[] = [];
		const seenByHook: unknown[] = [];
		const model: FakeModel = { samplingParams: { venice_parameters: { enable_web_search: "on" } } };
		const api = withVeniceRequestDefaults(fakeApi(sent), identityClamp);

		await simpleCall(api, model, {
			sessionId: "session-1",
			onPayload: (payload) => {
				seenByHook.push(payload);
				return { ...(payload as object), hooked: true };
			},
		});

		const expected = {
			...body(),
			venice_parameters: { include_venice_system_prompt: false, enable_web_search: "on" },
			prompt_cache_key: "session-1",
		};
		assert.deepEqual(seenByHook, [expected]);
		assert.deepEqual(sent, [{ ...expected, hooked: true }]);
	});

	it("applies defaults when the caller has no onPayload", async () => {
		const sent: unknown[] = [];
		await streamCall(withVeniceRequestDefaults(fakeApi(sent), identityClamp), {});
		assert.deepEqual(sent, [{ ...body(), venice_parameters: { include_venice_system_prompt: false } }]);
	});

	it("merges model, thinking-level, and request venice_parameters key by key", async () => {
		const sent: unknown[] = [];
		const model: FakeModel = {
			samplingParams: { venice_parameters: { enable_web_search: "on", strip_thinking_response: false } },
			samplingParamsByThinkingLevel: { high: { venice_parameters: { strip_thinking_response: true } } },
		};
		await simpleCall(withVeniceRequestDefaults(fakeApi(sent), identityClamp), model, {
			reasoning: "high",
			samplingParams: { venice_parameters: { enable_web_citations: true } },
		});
		assert.deepEqual(veniceParametersSent(sent), {
			include_venice_system_prompt: false,
			enable_web_search: "on",
			strip_thinking_response: true,
			enable_web_citations: true,
		});
	});

	it("lets a later layer override a key, including the default", async () => {
		const sent: unknown[] = [];
		const model: FakeModel = {
			samplingParams: { venice_parameters: { enable_web_search: "on" } },
			samplingParamsByThinkingLevel: {
				off: { venice_parameters: { enable_web_search: "off", include_venice_system_prompt: true } },
			},
		};
		await simpleCall(withVeniceRequestDefaults(fakeApi(sent), identityClamp), model, {});
		assert.deepEqual(veniceParametersSent(sent), { include_venice_system_prompt: true, enable_web_search: "off" });
	});

	it("uses the clamped thinking level and reasoningEffort on the stream path", async () => {
		const sent: unknown[] = [];
		const clamp = (_model: unknown, level: Level): Level => (level === "medium" ? "high" : level);
		const model: FakeModel = {
			samplingParams: { venice_parameters: { enable_web_search: "on" } },
			samplingParamsByThinkingLevel: { high: { venice_parameters: { strip_thinking_response: true } } },
		};
		await streamCall(withVeniceRequestDefaults(fakeApi(sent, clamp), clamp), model, {
			reasoningEffort: "medium",
		} as StreamOptions);
		assert.deepEqual(veniceParametersSent(sent), {
			include_venice_system_prompt: false,
			enable_web_search: "on",
			strip_thinking_response: true,
		});
	});

	it("disables thinking at level off where off has no wire value", async () => {
		const sent: unknown[] = [];
		const model: FakeModel = { reasoning: true, thinkingLevelMap: { off: undefined } };
		const api = withVeniceRequestDefaults(fakeApi(sent), identityClamp);

		await simpleCall(api, model, {});
		await streamCall(api, model);

		assert.deepEqual(veniceParametersSent(sent), {
			include_venice_system_prompt: false,
			disable_thinking: true,
		});
	});

	it("leaves thinking to reasoning_effort where off maps to none", async () => {
		const sent: unknown[] = [];
		const model: FakeModel = { reasoning: true, thinkingLevelMap: { off: "none" } };
		await simpleCall(withVeniceRequestDefaults(fakeApi(sent), identityClamp), model, {});
		assert.deepEqual(veniceParametersSent(sent), { include_venice_system_prompt: false });
	});

	it("leaves thinking alone above level off and on non-reasoning models", async () => {
		const sent: unknown[] = [];
		const api = withVeniceRequestDefaults(fakeApi(sent), identityClamp);

		await simpleCall(api, { reasoning: true }, { reasoning: "high" });
		await simpleCall(api, { reasoning: false }, {});

		for (const body of sent) assert.equal("disable_thinking" in (veniceParametersSent([body]) as object), false);
	});

	it("lets a user layer turn disable_thinking back off", async () => {
		const sent: unknown[] = [];
		const model: FakeModel = {
			reasoning: true,
			samplingParamsByThinkingLevel: { off: { venice_parameters: { disable_thinking: false } } },
		};
		await simpleCall(withVeniceRequestDefaults(fakeApi(sent), identityClamp), model, {});
		assert.deepEqual(veniceParametersSent(sent), {
			include_venice_system_prompt: false,
			disable_thinking: false,
		});
	});

	it("applies the off-level layer to a model whose off level is unset in its map", async () => {
		// Off is selectable without a thinkingLevelMap entry; clamping must keep it so the
		// user's off-level venice_parameters apply.
		const sent: unknown[] = [];
		const clamp = () => "off" as Level;
		const model: FakeModel = {
			reasoning: true,
			samplingParamsByThinkingLevel: { off: { venice_parameters: { disable_thinking: false } } },
		};
		await streamCall(withVeniceRequestDefaults(fakeApi(sent, clamp), clamp), model, {
			reasoningEffort: "high",
		} as StreamOptions);
		assert.equal((veniceParametersSent(sent) as Record<string, unknown>).disable_thinking, false);
	});
});

describe("usage and billed cost", () => {
	const model = {
		id: "claude-opus-5-5",
		api: "openai-completions",
		provider: "venice",
		cost: { input: 4, output: 20, cacheRead: 0.2, cacheWrite: 5 },
	};
	const clamp = ((_model: unknown, level: unknown) => level) as never;

	/**
	 * Minimal completions adapter: hands the raw usage chunk to `onProviderStreamEvent`, then ends
	 * with the message the adapter's own `parseChunkUsage` would produce from it.
	 */
	function usageApi(rawChunk: unknown, parsed: Usage): ProviderStreams {
		const run = (options?: StreamOptions) => {
			const stream = createAssistantMessageEventStream();
			void (async () => {
				// The adapter prices the tokens it parsed, as `parseChunkUsage` does.
				calculateCost(model as never, parsed);
				const message = assistant(parsed);
				stream.push({ type: "start", partial: { ...message, usage: zeroUsage(), stopReason: "pending" } });
				await options?.onProviderStreamEvent?.(rawChunk, model as never);
				stream.push({ type: "done", reason: "stop", message });
				stream.end();
			})();
			return stream;
		};
		return {
			stream: (_model, _context, options) => run(options),
			streamSimple: (_model, _context, options) => run(options as StreamOptions),
		};
	}

	const parsedUsage = (values: Partial<Usage>): Usage => ({
		...zeroUsage(),
		...values,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	});

	it("moves Venice's cache_creation_input_tokens onto cacheWrite and recomputes the cost", async () => {
		// Turn 1 measured against the live API: 15,637 prompt tokens, 15,621 written to cache.
		const parsed = parsedUsage({ input: 15637, output: 4, totalTokens: 15641 });
		const raw = {
			usage: {
				prompt_tokens: 15637,
				completion_tokens: 4,
				prompt_tokens_details: { cached_tokens: 0, cache_creation_input_tokens: 15621 },
				cache_creation_input_tokens: 15621,
			},
		};
		const message = await drain(withVeniceRequestDefaults(usageApi(raw, parsed), clamp).stream(model as never, {} as never));
		assert.equal(message?.usage.input, 16);
		assert.equal(message?.usage.cacheWrite, 15621);
		assert.equal(message?.usage.totalTokens, 15641);
		near(message!.usage.cost.input, (16 * 4) / 1e6);
		near(message!.usage.cost.cacheWrite, (15621 * 5) / 1e6);
		near(message!.usage.cost.total, (16 * 4 + 4 * 20 + 15621 * 5) / 1e6);
	});

	it("leaves cache reads alone when the turn only reads the cached prefix", async () => {
		const parsed = parsedUsage({ input: 35, output: 7, cacheRead: 15621, totalTokens: 15663 });
		const raw = { usage: { prompt_tokens: 15656, prompt_tokens_details: { cached_tokens: 15621 } } };
		const message = await drain(withVeniceRequestDefaults(usageApi(raw, parsed), clamp).stream(model as never, {} as never));
		assert.equal(message?.usage.input, 35);
		assert.equal(message?.usage.cacheWrite, 0);
		near(message!.usage.cost.cacheRead, (15621 * 0.2) / 1e6);
	});

	it("does not double count a provider that already reports cache_write_tokens", async () => {
		const parsed = parsedUsage({ input: 9900, output: 1, cacheWrite: 100, totalTokens: 10001 });
		const raw = {
			usage: { prompt_tokens: 10000, prompt_tokens_details: { cached_tokens: 0, cache_write_tokens: 100 } },
		};
		const message = await drain(withVeniceRequestDefaults(usageApi(raw, parsed), clamp).stream(model as never, {} as never));
		assert.equal(message?.usage.input, 9900);
		assert.equal(message?.usage.cacheWrite, 100);
	});

	const run = (raw: unknown, parsed: Usage) =>
		drain(withVeniceRequestDefaults(usageApi(raw, parsed), clamp).stream(model as never, {} as never));

	it("reports Venice's billed cost as the total and keeps the per-category estimates", async () => {
		// A web search turn: Venice bills about $0.01 for the search on top of the tokens.
		const parsed = parsedUsage({ input: 3369, output: 48, totalTokens: 3417 });
		const tokens = (3369 * 4 + 48 * 20) / 1e6;
		const raw = { usage: { prompt_tokens: 3369, completion_tokens: 48 }, cost: { usd: 0, diem: tokens + 0.01 } };
		const message = await run(raw, parsed);

		near(message!.usage.cost.total, tokens + 0.01);
		assert.equal(message!.usage.cost.totalFromProvider, true);
		near(message!.usage.cost.input, (3369 * 4) / 1e6);
		near(message!.usage.cost.output, (48 * 20) / 1e6);
	});

	it("applies the billed total after moving cache writes", async () => {
		const parsed = parsedUsage({ input: 15637, output: 4, totalTokens: 15641 });
		const billed = (16 * 4 + 4 * 20 + 15621 * 5) / 1e6;
		const raw = {
			usage: {
				prompt_tokens: 15637,
				completion_tokens: 4,
				prompt_tokens_details: { cached_tokens: 0, cache_creation_input_tokens: 15621 },
			},
			cost: { usd: 0, diem: billed },
		};
		const message = await run(raw, parsed);

		assert.equal(message!.usage.input, 16);
		assert.equal(message!.usage.cacheWrite, 15621);
		near(message!.usage.cost.cacheWrite, (15621 * 5) / 1e6);
		near(message!.usage.cost.total, billed);
		assert.equal(message!.usage.cost.totalFromProvider, true);
	});

	it("adds the USD and DIEM parts of a split payment", async () => {
		const parsed = parsedUsage({ input: 100, output: 10, totalTokens: 110 });
		const message = await run({ usage: { prompt_tokens: 100 }, cost: { usd: 0.002, diem: 0.0005 } }, parsed);
		near(message!.usage.cost.total, 0.0025);
		assert.equal(message!.usage.cost.totalFromProvider, true);
	});

	it("trusts a billed cost of zero", async () => {
		const parsed = parsedUsage({ input: 100, output: 10, totalTokens: 110 });
		const message = await run({ usage: { prompt_tokens: 100 }, cost: { usd: 0, diem: 0 } }, parsed);
		assert.equal(message!.usage.cost.total, 0);
		assert.equal(message!.usage.cost.totalFromProvider, true);
	});

	it("keeps the catalog estimate when the cost is missing or malformed", async () => {
		const estimate = (100 * 4 + 10 * 20) / 1e6;
		for (const cost of [undefined, null, {}, { usd: -1, diem: 0 }, { usd: "0.1", diem: 0 }, { usd: 0, diem: Number.NaN }, 0.5]) {
			const parsed = parsedUsage({ input: 100, output: 10, totalTokens: 110 });
			const raw = cost === undefined ? { usage: { prompt_tokens: 100 } } : { usage: { prompt_tokens: 100 }, cost };
			const message = await run(raw, parsed);
			near(message!.usage.cost.total, estimate);
			assert.notEqual(message!.usage.cost.totalFromProvider, true, `cost ${JSON.stringify(cost)}`);
		}
	});

	it("still passes every provider chunk to the caller's onProviderStreamEvent", async () => {
		const seen: unknown[] = [];
		const raw = { usage: { prompt_tokens: 10, prompt_tokens_details: { cache_creation_input_tokens: 8 } } };
		const parsed = parsedUsage({ input: 10, output: 1, totalTokens: 11 });
		await drain(
			withVeniceRequestDefaults(usageApi(raw, parsed), clamp).stream(model as never, {} as never, {
				onProviderStreamEvent: (event) => {
					seen.push(event);
				},
			}),
		);
		assert.deepEqual(seen, [raw]);
	});
});
