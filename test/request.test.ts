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
	/** Fake API that builds a body the way pi does, then sends it through options.onPayload. */
	function fakeApi(sent: unknown[], userBody: Record<string, unknown>): ProviderStreams {
		const run = async (options?: StreamOptions) => {
			const replaced = await options?.onPayload?.(userBody, {} as never);
			sent.push(replaced === undefined ? userBody : replaced);
		};
		return {
			stream: (_model, _context, options) => run(options) as never,
			streamSimple: (_model, _context, options) => run(options) as never,
		};
	}

	it("applies defaults before the caller's onPayload, which can still replace the body", async () => {
		const sent: unknown[] = [];
		const seenByHook: unknown[] = [];
		const api = withVeniceRequestDefaults(fakeApi(sent, body({ venice_parameters: { enable_web_search: "on" } })));

		await api.streamSimple({} as never, {} as never, {
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
		await withVeniceRequestDefaults(fakeApi(sent, body())).stream({} as never, {} as never, undefined);
		assert.deepEqual(sent, [{ ...body(), venice_parameters: { include_venice_system_prompt: false } }]);
	});
});
