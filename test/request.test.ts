import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import type { ProviderStreams, StreamOptions } from "@earendil-works/pi-ai";
import { applyVeniceDefaults, withVeniceRequestDefaults } from "../extensions/venice/request.ts";

const body = (extra: Record<string, unknown> = {}) => ({ model: "kimi-k3", messages: [], ...extra });

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
		return {
			stream: (model, _context, options) =>
				run(model as FakeModel, (options as { reasoningEffort?: Level })?.reasoningEffort ?? "off", options) as never,
			streamSimple: (model, _context, options) => run(model as FakeModel, options?.reasoning ?? "off", options) as never,
		};
	}

	const veniceParametersSent = (sent: unknown[]) => (sent[0] as Record<string, unknown>).venice_parameters;

	it("applies defaults before the caller's onPayload, which can still replace the body", async () => {
		const sent: unknown[] = [];
		const seenByHook: unknown[] = [];
		const model: FakeModel = { samplingParams: { venice_parameters: { enable_web_search: "on" } } };
		const api = withVeniceRequestDefaults(fakeApi(sent), identityClamp);

		await api.streamSimple(model as never, {} as never, {
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
		await withVeniceRequestDefaults(fakeApi(sent), identityClamp).stream({} as never, {} as never, undefined);
		assert.deepEqual(sent, [{ ...body(), venice_parameters: { include_venice_system_prompt: false } }]);
	});

	it("merges model, thinking-level, and request venice_parameters key by key", async () => {
		const sent: unknown[] = [];
		const model: FakeModel = {
			samplingParams: { venice_parameters: { enable_web_search: "on", strip_thinking_response: false } },
			samplingParamsByThinkingLevel: { high: { venice_parameters: { strip_thinking_response: true } } },
		};
		await withVeniceRequestDefaults(fakeApi(sent), identityClamp).streamSimple(model as never, {} as never, {
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
		await withVeniceRequestDefaults(fakeApi(sent), identityClamp).streamSimple(model as never, {} as never, {});
		assert.deepEqual(veniceParametersSent(sent), { include_venice_system_prompt: true, enable_web_search: "off" });
	});

	it("uses the clamped thinking level and reasoningEffort on the stream path", async () => {
		const sent: unknown[] = [];
		const clamp = (_model: unknown, level: Level): Level => (level === "medium" ? "high" : level);
		const model: FakeModel = {
			samplingParams: { venice_parameters: { enable_web_search: "on" } },
			samplingParamsByThinkingLevel: { high: { venice_parameters: { strip_thinking_response: true } } },
		};
		await withVeniceRequestDefaults(fakeApi(sent, clamp), clamp).stream(model as never, {} as never, {
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

		await api.streamSimple(model as never, {} as never, {});
		await api.stream(model as never, {} as never, undefined);

		assert.deepEqual(veniceParametersSent(sent), {
			include_venice_system_prompt: false,
			disable_thinking: true,
		});
	});

	it("leaves thinking to reasoning_effort where off maps to none", async () => {
		const sent: unknown[] = [];
		const model: FakeModel = { reasoning: true, thinkingLevelMap: { off: "none" } };
		await withVeniceRequestDefaults(fakeApi(sent), identityClamp).streamSimple(model as never, {} as never, {});
		assert.deepEqual(veniceParametersSent(sent), { include_venice_system_prompt: false });
	});

	it("leaves thinking alone above level off and on non-reasoning models", async () => {
		const sent: unknown[] = [];
		const api = withVeniceRequestDefaults(fakeApi(sent), identityClamp);

		await api.streamSimple({ reasoning: true } as never, {} as never, { reasoning: "high" });
		await api.streamSimple({ reasoning: false } as never, {} as never, {});

		for (const body of sent) assert.equal("disable_thinking" in (veniceParametersSent([body]) as object), false);
	});

	it("lets a user layer turn disable_thinking back off", async () => {
		const sent: unknown[] = [];
		const model: FakeModel = {
			reasoning: true,
			samplingParamsByThinkingLevel: { off: { venice_parameters: { disable_thinking: false } } },
		};
		await withVeniceRequestDefaults(fakeApi(sent), identityClamp).streamSimple(model as never, {} as never, {});
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
		await withVeniceRequestDefaults(fakeApi(sent, clamp), clamp).stream(model as never, {} as never, {
			reasoningEffort: "high",
		} as StreamOptions);
		assert.equal((veniceParametersSent(sent) as Record<string, unknown>).disable_thinking, false);
	});
});
