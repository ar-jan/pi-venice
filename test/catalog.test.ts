import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { toPiModel } from "../extensions/venice/index.ts";

type Capabilities = Record<string, unknown>;

function veniceModel(capabilities: Capabilities = {}, extra: Record<string, unknown> = {}) {
	return {
		id: "test-model",
		type: "text",
		model_spec: {
			name: "Test Model",
			privacy: "private",
			availableContextTokens: 128_000,
			maxCompletionTokens: 16_384,
			capabilities: { supportsFunctionCalling: true, ...capabilities },
			pricing: { input: { usd: 1 }, output: { usd: 2 } },
			...extra,
		},
	};
}

describe("toPiModel", () => {
	it("omits E2EE models, which this provider cannot serve", () => {
		assert.equal(toPiModel(veniceModel({ supportsE2EE: true }) as never), undefined);
	});

	it("keeps models that support tool calling", () => {
		const model = toPiModel(veniceModel() as never);
		assert.equal(model?.id, "test-model");
		assert.equal(model?.name, "Test Model (private)");
	});

	it("marks TEE attestation without E2EE", () => {
		const model = toPiModel(veniceModel({ supportsTeeAttestation: true }) as never);
		assert.equal(model?.name, "Test Model (private, TEE)");
	});
});
