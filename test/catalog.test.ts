import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { getSupportedThinkingLevels, InMemoryCredentialStore, InMemoryModelsStore, type Provider } from "@earendil-works/pi-ai";
import { ModelRegistry, ModelRuntime } from "@earendil-works/pi-coding-agent";
import registerVenice, {
	createVeniceProvider,
	DECISION_MODELS_TIMEOUT_MS,
	toPiClassifierModel,
	toPiModel,
} from "../extensions/venice/index.ts";

type Capabilities = Record<string, unknown>;

function veniceModel(
	capabilities: Capabilities = {},
	id = "grok-4-20",
	limits: { contextWindow?: number; maxTokens?: number } = {},
) {
	return {
		id,
		type: "text",
		model_spec: {
			name: "Test Model",
			privacy: "private",
			availableContextTokens: limits.contextWindow ?? 128_000,
			maxCompletionTokens: limits.maxTokens ?? 16_384,
			capabilities: { supportsFunctionCalling: true, ...capabilities },
			pricing: { input: { usd: 1 }, output: { usd: 2 } },
		},
	};
}

function veniceDecisionModel(
	id = "jev-latest",
	overrides: Record<string, unknown> = {},
	type: string | undefined = "decision",
) {
	return {
		id,
		...(type === undefined ? {} : { type }),
		model_spec: {
			name: id === "jev-latest" ? "Jev (System One)" : "Clef (System One)",
			privacy: id === "jev-latest" ? "anonymized" : "private",
			offline: false,
			maxStateTokens: 32_000,
			maxTotalTokens: id === "jev-latest" ? 64_000 : 32_768,
			pricing: { input: { usd: 0.042 }, output: { usd: 0 } },
			...overrides,
		},
	};
}

describe("catalog refresh", () => {
	it("uses the provider base URL from models.json for catalog discovery", async (t) => {
		const agentDir = await mkdtemp(join(tmpdir(), "pi-venice-catalog-"));
		t.after(() => rm(agentDir, { recursive: true, force: true }));
		const modelsPath = join(agentDir, "models.json");
		const baseUrl = "https://gateway.example/venice/v1/";
		await writeFile(modelsPath, JSON.stringify({ providers: { venice: { baseUrl } } }));

		const credentials = new InMemoryCredentialStore();
		await credentials.modify("venice", async () => ({ type: "api_key", key: "gateway-key" }));
		const runtime = await ModelRuntime.create({ credentials, modelsPath });
		const registry = new ModelRegistry(runtime);
		const refresh = t.mock.method(runtime, "refresh");
		registerVenice({
			on() {},
			registerCommand() {},
			registerProvider(provider: Provider) {
				registry.registerProvider(provider);
			},
		} as never);
		// Let the cache-only refresh triggered by registration finish before refreshing over the network.
		await refresh.mock.calls[0]?.result;
		const fetch = t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => {
			const url = String(input);
			if (url.includes("type=decision")) return Response.json({ data: [veniceDecisionModel("jev-latest")] });
			return Response.json({ data: [veniceModel()] });
		});

		const result = await registry.refresh({ providers: ["venice"], allowNetwork: true });

		assert.equal(result.errors.size, 0);
		assert.equal(fetch.mock.callCount(), 2);
		const urls = fetch.mock.calls.map((call) => String(call.arguments[0])).sort();
		assert.deepEqual(urls, [
			"https://gateway.example/venice/v1/models?type=decision",
			"https://gateway.example/venice/v1/models?type=text",
		]);
		for (const call of fetch.mock.calls) {
			assert.equal(new Headers(call.arguments[1]?.headers).get("authorization"), "Bearer gateway-key");
		}
		assert.equal(registry.find("venice", "grok-4-20")?.baseUrl, baseUrl);
		assert.equal(registry.findOfType("classifier", "venice", "jev-latest")?.baseUrl, baseUrl);
	});

	it("keeps chat models refreshing and preserves cached classifiers when the decision list fails", async (t) => {
		const credentials = new InMemoryCredentialStore();
		await credentials.modify("venice", async () => ({ type: "api_key", key: "test-key" }));
		const modelsStore = new InMemoryModelsStore();
		const cachedDecision = toPiClassifierModel(veniceDecisionModel("jev-latest") as never)!;
		await modelsStore.write("venice", { models: [cachedDecision], checkedAt: 1 });
		const runtime = await ModelRuntime.create({ credentials, modelsStore, modelsPath: null });
		const registry = new ModelRegistry(runtime);
		const refresh = t.mock.method(runtime, "refresh");
		registerVenice({
			on() {},
			registerCommand() {},
			registerProvider(provider: Provider) {
				registry.registerProvider(provider);
			},
		} as never);
		await refresh.mock.calls[0]?.result;

		t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => {
			if (String(input).includes("type=decision")) return new Response("unavailable", { status: 503 });
			return Response.json({ data: [veniceModel()] });
		});

		const result = await registry.refresh({ providers: ["venice"], allowNetwork: true });

		assert.equal(result.errors.size, 0);
		assert.equal(registry.find("venice", "grok-4-20")?.id, "grok-4-20");
		assert.equal(registry.findOfType("classifier", "venice", "jev-latest")?.id, "jev-latest");
	});

	it("still loads chat models when the decision list fails with an empty cache", async (t) => {
		const credentials = new InMemoryCredentialStore();
		await credentials.modify("venice", async () => ({ type: "api_key", key: "test-key" }));
		const runtime = await ModelRuntime.create({ credentials, modelsStore: new InMemoryModelsStore(), modelsPath: null });
		const registry = new ModelRegistry(runtime);
		const refresh = t.mock.method(runtime, "refresh");
		registerVenice({
			on() {},
			registerCommand() {},
			registerProvider(provider: Provider) {
				registry.registerProvider(provider);
			},
		} as never);
		await refresh.mock.calls[0]?.result;

		t.mock.method(globalThis, "fetch", async (input: string | URL | Request) => {
			if (String(input).includes("type=decision")) throw new Error("network down");
			return Response.json({ data: [veniceModel()] });
		});

		const result = await registry.refresh({ providers: ["venice"], allowNetwork: true });

		assert.equal(result.errors.size, 0);
		assert.equal(registry.find("venice", "grok-4-20")?.id, "grok-4-20");
		assert.deepEqual(registry.getModelsOfType("classifier", "venice"), []);
	});

	it("publishes and persists fresh chat models when the decision endpoint stalls within Pi's 15s deadline", async (t) => {
		const credentials = new InMemoryCredentialStore();
		await credentials.modify("venice", async () => ({ type: "api_key", key: "test-key" }));
		const modelsStore = new InMemoryModelsStore();
		const runtime = await ModelRuntime.create({ credentials, modelsStore, modelsPath: null });
		const registry = new ModelRegistry(runtime);
		const initialRefresh = t.mock.method(runtime, "refresh");
		registerVenice({
			on() {},
			registerCommand() {},
			registerProvider(provider: Provider) {
				registry.registerProvider(provider);
			},
		} as never);
		await initialRefresh.mock.calls[0]?.result;

		t.mock.timers.enable({ apis: ["setTimeout"] });
		let decisionStarted!: () => void;
		const decisionPending = new Promise<void>((resolve) => {
			decisionStarted = resolve;
		});
		let decisionAborted = false;
		t.mock.method(globalThis, "fetch", (input: string | URL | Request, init?: RequestInit) => {
			if (String(input).includes("type=decision")) {
				decisionStarted();
				return new Promise<Response>((_resolve, reject) => {
					init?.signal?.addEventListener(
						"abort",
						() => {
							decisionAborted = true;
							reject(init.signal!.reason ?? new DOMException("Aborted", "AbortError"));
						},
						{ once: true },
					);
				});
			}
			return Promise.resolve(Response.json({ data: [veniceModel()] }));
		});

		// Mirror Pi's interactive refresh: the 15s caller deadline starts before refresh() runs.
		const interactive = new AbortController();
		const interactiveTimeout = setTimeout(() => interactive.abort(new DOMException("Timed out", "TimeoutError")), 15_000);
		t.after(() => clearTimeout(interactiveTimeout));

		const refreshPromise = registry.refresh({ providers: ["venice"], allowNetwork: true, signal: interactive.signal });
		await decisionPending;
		// Advance to the decision timeout — still well before the 15s interactive deadline.
		assert.ok(DECISION_MODELS_TIMEOUT_MS < 15_000);
		t.mock.timers.tick(DECISION_MODELS_TIMEOUT_MS);

		const result = await refreshPromise;
		assert.equal(interactive.signal.aborted, false);
		assert.equal(decisionAborted, true);
		assert.equal(result.aborted, false);
		assert.equal(result.errors.size, 0);
		assert.equal(registry.find("venice", "grok-4-20")?.id, "grok-4-20");
		const persisted = await modelsStore.read("venice");
		assert.deepEqual(
			persisted?.models.map((entry) => entry.id),
			["grok-4-20"],
		);
	});

	it("falls back to cached classifiers and persists fresh chats when the decision endpoint stalls", async (t) => {
		const credentials = new InMemoryCredentialStore();
		await credentials.modify("venice", async () => ({ type: "api_key", key: "test-key" }));
		const modelsStore = new InMemoryModelsStore();
		const cachedDecision = toPiClassifierModel(veniceDecisionModel("jev-latest") as never)!;
		await modelsStore.write("venice", { models: [cachedDecision], checkedAt: 1 });
		const runtime = await ModelRuntime.create({ credentials, modelsStore, modelsPath: null });
		const registry = new ModelRegistry(runtime);
		const initialRefresh = t.mock.method(runtime, "refresh");
		registerVenice({
			on() {},
			registerCommand() {},
			registerProvider(provider: Provider) {
				registry.registerProvider(provider);
			},
		} as never);
		await initialRefresh.mock.calls[0]?.result;

		t.mock.timers.enable({ apis: ["setTimeout"] });
		let decisionStarted!: () => void;
		const decisionPending = new Promise<void>((resolve) => {
			decisionStarted = resolve;
		});
		t.mock.method(globalThis, "fetch", (input: string | URL | Request, init?: RequestInit) => {
			if (String(input).includes("type=decision")) {
				decisionStarted();
				return new Promise<Response>((_resolve, reject) => {
					init?.signal?.addEventListener(
						"abort",
						() => reject(init.signal!.reason ?? new DOMException("Aborted", "AbortError")),
						{ once: true },
					);
				});
			}
			return Promise.resolve(Response.json({ data: [veniceModel()] }));
		});

		const interactive = new AbortController();
		const interactiveTimeout = setTimeout(() => interactive.abort(), 15_000);
		t.after(() => clearTimeout(interactiveTimeout));

		const refreshPromise = registry.refresh({ providers: ["venice"], allowNetwork: true, signal: interactive.signal });
		await decisionPending;
		t.mock.timers.tick(DECISION_MODELS_TIMEOUT_MS);

		const result = await refreshPromise;
		assert.equal(result.aborted, false);
		assert.equal(result.errors.size, 0);
		assert.equal(registry.find("venice", "grok-4-20")?.id, "grok-4-20");
		assert.equal(registry.findOfType("classifier", "venice", "jev-latest")?.id, "jev-latest");
		const persisted = await modelsStore.read("venice");
		assert.deepEqual(
			persisted?.models.map((entry) => entry.id),
			["grok-4-20", "jev-latest"],
		);
	});

	it("preserves caller cancellation while waiting on a stalled decision endpoint", async (t) => {
		const credentials = new InMemoryCredentialStore();
		await credentials.modify("venice", async () => ({ type: "api_key", key: "test-key" }));
		const modelsStore = new InMemoryModelsStore();
		const runtime = await ModelRuntime.create({ credentials, modelsStore, modelsPath: null });
		const registry = new ModelRegistry(runtime);
		const initialRefresh = t.mock.method(runtime, "refresh");
		registerVenice({
			on() {},
			registerCommand() {},
			registerProvider(provider: Provider) {
				registry.registerProvider(provider);
			},
		} as never);
		await initialRefresh.mock.calls[0]?.result;

		let decisionStarted!: () => void;
		const decisionPending = new Promise<void>((resolve) => {
			decisionStarted = resolve;
		});
		t.mock.method(globalThis, "fetch", (input: string | URL | Request, init?: RequestInit) => {
			if (String(input).includes("type=decision")) {
				decisionStarted();
				return new Promise<Response>((_resolve, reject) => {
					init?.signal?.addEventListener(
						"abort",
						() => reject(init.signal!.reason ?? new DOMException("Aborted", "AbortError")),
						{ once: true },
					);
				});
			}
			return Promise.resolve(Response.json({ data: [veniceModel()] }));
		});

		const caller = new AbortController();
		const refreshPromise = registry.refresh({ providers: ["venice"], allowNetwork: true, signal: caller.signal });
		await decisionPending;
		caller.abort(new DOMException("Cancelled by caller", "AbortError"));

		const result = await refreshPromise;
		assert.equal(result.aborted, true);
		assert.equal(registry.find("venice", "grok-4-20"), undefined);
		assert.equal(await modelsStore.read("venice"), undefined);
	});

	it("excludes cached classifiers after a cache-only restore when TypeSafe is unavailable", async () => {
		const credentials = new InMemoryCredentialStore();
		await credentials.modify("venice", async () => ({ type: "api_key", key: "test-key" }));
		const modelsStore = new InMemoryModelsStore();
		const cachedChat = toPiModel(veniceModel() as never)!;
		const cachedDecision = toPiClassifierModel(veniceDecisionModel("jev-latest") as never)!;
		await modelsStore.write("venice", { models: [cachedChat, cachedDecision], checkedAt: 1 });

		const runtime = await ModelRuntime.create({ credentials, modelsStore, modelsPath: null });
		const registry = new ModelRegistry(runtime);
		registry.registerProvider(createVeniceProvider({ systemOne: undefined }));

		const result = await registry.refresh({ providers: ["venice"], allowNetwork: false });

		assert.equal(result.errors.size, 0);
		assert.deepEqual(registry.getModelsOfType("classifier", "venice"), []);
		assert.equal(registry.findOfType("classifier", "venice", "jev-latest"), undefined);
		assert.deepEqual(await registry.getAvailableOfType("classifier", "venice"), []);
		assert.equal(registry.find("venice", "grok-4-20")?.id, "grok-4-20");
	});
});

describe("toPiClassifierModel", () => {
	it("maps jev models to assets compat without the opt-in header", () => {
		const model = toPiClassifierModel(veniceDecisionModel("jev-latest") as never);
		assert.deepEqual(model, {
			type: "classifier",
			id: "jev-latest",
			name: "Jev (System One) (anonymized)",
			api: "typesafe-system-one",
			provider: "venice",
			baseUrl: "https://api.venice.ai/api/v1",
			input: ["text", "image"],
			cost: { input: 0.042, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 64_000,
			compat: { imageInput: "assets", sendImageOptInHeader: false },
		});
	});

	it("maps clef and other decision models to images compat without the opt-in header", () => {
		const model = toPiClassifierModel(
			veniceDecisionModel("clef-api", {
				pricing: { input: { usd: 0.05 }, output: { usd: 0.05 } },
			}) as never,
		);
		assert.deepEqual(model, {
			type: "classifier",
			id: "clef-api",
			name: "Clef (System One) (private)",
			api: "typesafe-system-one",
			provider: "venice",
			baseUrl: "https://api.venice.ai/api/v1",
			input: ["text", "image"],
			cost: { input: 0.05, output: 0.05, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 32_768,
			compat: { imageInput: "images", sendImageOptInHeader: false },
		});
	});

	it("honors explicit supportsVision: false in capabilities", () => {
		const model = toPiClassifierModel(
			veniceDecisionModel("clef-text-only", { capabilities: { supportsVision: false } }) as never,
		);
		assert.deepEqual(model?.input, ["text"]);
	});

	it("skips offline, removed, and non-decision entries", () => {
		assert.equal(toPiClassifierModel(veniceDecisionModel("jev-latest", { offline: true }) as never), undefined);
		assert.equal(
			toPiClassifierModel(
				veniceDecisionModel("jev-latest", { deprecation: { removesAt: "2020-01-01T00:00:00Z" } }) as never,
			),
			undefined,
		);
		assert.equal(toPiClassifierModel(veniceDecisionModel("grok-4-20", {}, "text") as never), undefined);
	});
});

describe("toPiModel", () => {
	it("keeps E2EE-capable models and labels them TEE, not E2EE", () => {
		// Without E2EE headers Venice runs them in TEE-only mode, so plain requests work.
		const model = toPiModel(
			veniceModel({ supportsE2EE: true, supportsTeeAttestation: true }, "e2ee-glm-5-3-flash") as never,
		);
		assert.equal(model?.id, "e2ee-glm-5-3-flash");
		assert.equal(model?.name, "Test Model (private, TEE)");
		assert.doesNotMatch(model!.name, /E2EE/i);
	});

	it("omits a model whose tool calling is verified broken", () => {
		const entry = veniceModel({ supportsE2EE: true, supportsTeeAttestation: true, supportsVision: true }, "e2ee-qwen3-vl-30b-a3b-p");
		assert.equal(toPiModel(entry as never), undefined);
	});

	it("does not offer off via a listed none effort in TEE-only mode", () => {
		// Venice rejects every way of disabling reasoning on these models with HTTP 400.
		const model = toPiModel(
			veniceModel(
				{
					supportsE2EE: true,
					supportsTeeAttestation: true,
					supportsReasoning: true,
					supportsReasoningEffort: true,
					reasoningEffortOptions: ["none", "low", "medium", "high"],
				},
				"e2ee-glm-5-3-p",
			) as never,
		);
		assert.equal(model?.thinkingLevelMap?.off, null);
		assert.deepEqual(getSupportedThinkingLevels(model!), ["low", "medium", "high"]);
	});

	it("keeps models that support tool calling", () => {
		const model = toPiModel(veniceModel() as never);
		assert.equal(model?.id, "grok-4-20");
		assert.equal(model?.name, "Test Model (private)");
	});

	it("marks TEE attestation without E2EE", () => {
		const model = toPiModel(veniceModel({ supportsTeeAttestation: true }) as never);
		assert.equal(model?.name, "Test Model (private, TEE)");
	});

	it("offers off without a map entry on a model verified to honor disable_thinking", () => {
		const model = toPiModel(
			veniceModel({ supportsReasoning: true, supportsReasoningEffort: true, reasoningEffortOptions: ["low", "high"] }) as never,
		);
		assert.equal(model?.reasoning, true);
		assert.equal(model?.thinkingLevelMap?.off, undefined);
		assert.equal(model?.thinkingLevelMap?.low, "low");
		assert.equal(model?.thinkingLevelMap?.medium, null);
		assert.deepEqual(getSupportedThinkingLevels(model!), ["off", "low", "high"]);
	});

	it("keeps off selectable on a verified model without effort control", () => {
		const model = toPiModel(veniceModel({ supportsReasoning: true, supportsReasoningEffort: false }) as never);
		assert.equal(model?.reasoning, true);
		assert.equal(model?.thinkingLevelMap?.off, undefined);
		assert.equal(model?.compat?.supportsReasoningEffort, false);
		// Only "medium" (model default) joins off: low/high/xhigh/max stay unsupported.
		assert.deepEqual(getSupportedThinkingLevels(model!), ["off", "medium"]);
	});

	it("does not offer off on a model that only strips reasoning", () => {
		// Same capabilities, but not on the verified list: Venice would only hide the reasoning.
		const model = toPiModel(
			veniceModel(
				{ supportsReasoning: true, supportsReasoningEffort: true, reasoningEffortOptions: ["low", "high"] },
				"unverified-model",
			) as never,
		);
		assert.equal(model?.thinkingLevelMap?.off, null);
		assert.deepEqual(getSupportedThinkingLevels(model!), ["low", "high"]);
	});

	it("uses none for off on a model verified to honor an unlisted none effort", () => {
		const model = toPiModel(
			veniceModel(
				{ supportsReasoning: true, supportsReasoningEffort: true, reasoningEffortOptions: ["low", "medium", "high"] },
				"kimi-k2-7-code",
			) as never,
		);
		assert.equal(model?.thinkingLevelMap?.off, "none");
		assert.equal(model?.compat?.supportsReasoningEffort, true);
		assert.deepEqual(getSupportedThinkingLevels(model!), ["off", "low", "medium", "high"]);
	});

	it("does not offer an unlisted none where pi would send no effort", () => {
		// Without effort control pi sends no reasoning_effort, so "off" would change nothing.
		const model = toPiModel(veniceModel({ supportsReasoning: true, supportsReasoningEffort: false }, "kimi-k2-7-code") as never);
		assert.equal(model?.thinkingLevelMap?.off, null);
		assert.deepEqual(getSupportedThinkingLevels(model!), ["medium"]);
	});

	it("uses none for off when the catalog lists that effort", () => {
		const model = toPiModel(
			veniceModel({ supportsReasoning: true, supportsReasoningEffort: true, reasoningEffortOptions: ["none", "high"] }) as never,
		);
		assert.equal(model?.thinkingLevelMap?.off, "none");
		assert.deepEqual(getSupportedThinkingLevels(model!), ["off", "high"]);
	});

	it("clamps a degenerate output cap instead of passing it through", () => {
		const equal = toPiModel(veniceModel({}, "equal", { contextWindow: 128_000, maxTokens: 128_000 }) as never);
		assert.equal(equal?.contextWindow, 128_000);
		assert.equal(equal?.maxTokens, 32_000);

		const above = toPiModel(veniceModel({}, "above", { contextWindow: 64_000, maxTokens: 128_000 }) as never);
		assert.equal(above?.contextWindow, 64_000);
		assert.equal(above?.maxTokens, 16_000);
	});

	it("keeps a large but valid output cap", () => {
		const model = toPiModel(veniceModel({}, "large", { contextWindow: 500_000, maxTokens: 200_000 }) as never);
		assert.equal(model?.maxTokens, 200_000);
	});

	it("uses the developer role Venice supports on reasoning models", () => {
		const model = toPiModel(veniceModel({ supportsReasoning: true }) as never);
		assert.equal(model?.compat?.supportsDeveloperRole, true);
	});

	it("leaves non-reasoning models without a thinking level map", () => {
		const model = toPiModel(veniceModel({ supportsReasoning: false }) as never);
		assert.equal(model?.reasoning, false);
		assert.equal(model?.thinkingLevelMap, undefined);
	});
});
