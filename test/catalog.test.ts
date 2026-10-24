import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { getSupportedThinkingLevels } from "@earendil-works/pi-ai";
import { toPiModel } from "../extensions/venice/index.ts";

type Capabilities = Record<string, unknown>;

function veniceModel(capabilities: Capabilities = {}, id = "grok-4-20") {
	return {
		id,
		type: "text",
		model_spec: {
			name: "Test Model",
			privacy: "private",
			availableContextTokens: 128_000,
			maxCompletionTokens: 16_384,
			capabilities: { supportsFunctionCalling: true, ...capabilities },
			pricing: { input: { usd: 1 }, output: { usd: 2 } },
		},
	};
}

describe("toPiModel", () => {
	it("omits E2EE models, which this provider cannot serve", () => {
		assert.equal(toPiModel(veniceModel({ supportsE2EE: true }) as never), undefined);
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

	it("uses none for off when the catalog lists that effort", () => {
		const model = toPiModel(
			veniceModel({ supportsReasoning: true, supportsReasoningEffort: true, reasoningEffortOptions: ["none", "high"] }) as never,
		);
		assert.equal(model?.thinkingLevelMap?.off, "none");
		assert.deepEqual(getSupportedThinkingLevels(model!), ["off", "high"]);
	});

	it("leaves non-reasoning models without a thinking level map", () => {
		const model = toPiModel(veniceModel({ supportsReasoning: false }) as never);
		assert.equal(model?.reasoning, false);
		assert.equal(model?.thinkingLevelMap, undefined);
	});
});
