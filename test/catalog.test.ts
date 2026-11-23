import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { getSupportedThinkingLevels, InMemoryCredentialStore, type Provider } from "@earendil-works/pi-ai";
import { ModelRegistry, ModelRuntime } from "@earendil-works/pi-coding-agent";
import registerVenice, { toPiModel } from "../extensions/venice/index.ts";

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
		const fetch = t.mock.method(globalThis, "fetch", async () => Response.json({ data: [veniceModel()] }));

		const result = await registry.refresh({ providers: ["venice"], allowNetwork: true });

		assert.equal(result.errors.size, 0);
		assert.equal(fetch.mock.callCount(), 1);
		const [url, options] = fetch.mock.calls[0]!.arguments;
		assert.equal(String(url), "https://gateway.example/venice/v1/models?type=text");
		assert.equal(new Headers(options?.headers).get("authorization"), "Bearer gateway-key");
		assert.equal(registry.find("venice", "grok-4-20")?.baseUrl, baseUrl);
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
