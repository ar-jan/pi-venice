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
import { applyVeniceDefaults, withVeniceRequestDefaults } from "../extensions/venice/request.ts";

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

describe("withVeniceRequestDefaults", () => {
	type Level = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
	type FakeModel = {
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

	const streamCall = (api: ProviderStreams, model: FakeModel, options?: StreamOptions) =>
		drain(api.stream(model as never, {} as never, options));
	const simpleCall = (api: ProviderStreams, model: FakeModel, options?: SimpleStreamOptions) =>
		drain(api.streamSimple(model as never, {} as never, options));

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

describe("cache write usage", () => {
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
