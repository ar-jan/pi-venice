import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { ClassifierContext, ClassifierModel } from "@earendil-works/pi-ai";
import { borrowedSystemOne } from "../extensions/venice/classify.ts";
import { toPiClassifierModel } from "../extensions/venice/index.ts";
import { Keyring, withClassifierKeyRotation } from "../extensions/venice/keyring.ts";

const ALLOWED_TOP_LEVEL_KEYS = new Set(["model", "state", "questions", "assets", "images"]);

function decisionModel(id: string, overrides: Record<string, unknown> = {}): ClassifierModel<"typesafe-system-one"> {
	const model = toPiClassifierModel({
		id,
		type: "decision",
		model_spec: {
			name: id === "jev-latest" ? "Jev (System One)" : "Clef (System One)",
			privacy: id === "jev-latest" ? "anonymized" : "private",
			maxStateTokens: 32_000,
			maxTotalTokens: id === "jev-latest" ? 64_000 : 32_768,
			pricing: { input: { usd: 0.05 }, output: { usd: 0.05 } },
			...overrides,
		},
	});
	assert.ok(model);
	return model;
}

const boolContext: ClassifierContext = {
	state: { message: "Looks great, ship it." },
	questions: {
		approved: {
			type: "bool",
			instructions: "Does the message approve the change?",
			criteria: { true: "Approves", false: "Does not approve" },
		},
	},
};

const samplePng = { type: "image" as const, data: "iVBORw0KGgo=", mimeType: "image/png" };

describe("borrowedSystemOne", () => {
	it("returns a classifier on the pinned Pi version", () => {
		const classifier = borrowedSystemOne();
		assert.equal(typeof classifier?.classify, "function");
	});
});

describe("Venice System One payload guard", () => {
	for (const [id, expectedImageField] of [
		["jev-latest", "assets"],
		["clef-api", "images"],
	] as const) {
		it(`sends Venice-valid System One payloads for ${id} (with image and bool→noul)`, async () => {
			const classifier = borrowedSystemOne();
			assert.ok(classifier);
			const model = decisionModel(id);

			const requests: { url: string; headers: Headers; body: Record<string, unknown> }[] = [];
			const stubFetch: typeof globalThis.fetch = async (input, init) => {
				requests.push({
					url: String(input),
					headers: new Headers(init?.headers),
					body: JSON.parse(String(init?.body)) as Record<string, unknown>,
				});
				return Response.json({
					model: id,
					answers: { approved: { type: "noul", noul: 0.94 } },
					usage: { input_tokens: 100, output_tokens: 10 },
				});
			};

			const result = await classifier.classify(
				model,
				{ ...boolContext, images: [samplePng] },
				{ apiKey: "venice-key", fetch: stubFetch },
			);

			assert.equal(result.stopReason, "stop");
			assert.deepEqual(result.answers, { approved: { type: "bool", probability: 0.94 } });
			assert.equal(result.usage?.input, 100);
			assert.equal(result.usage?.output, 10);

			assert.equal(requests.length, 1);
			const req = requests[0]!;
			assert.equal(req.url, "https://api.venice.ai/api/v1/systemone");
			assert.equal(req.headers.get("authorization"), "Bearer venice-key");
			// Venice does not accept TypeSafe's experimental image opt-in header requirement.
			assert.equal(req.headers.get("typesafe-danger-zone-images"), null);

			// Venice's CreateDecisionRequestSchema sets additionalProperties: false.
			for (const key of Object.keys(req.body)) {
				assert.ok(ALLOWED_TOP_LEVEL_KEYS.has(key), `unexpected top-level key: ${key}`);
			}
			assert.equal(req.body.model, id);
			const questions = req.body.questions as Record<string, { type: string }>;
			assert.equal(questions.approved?.type, "noul");
			assert.ok(expectedImageField in req.body, `expected ${expectedImageField} in request body`);
			assert.equal((expectedImageField === "assets" ? "images" : "assets") in req.body, false);
		});
	}

	it("rotates keys end-to-end through the borrowed classifier on HTTP 402", async () => {
		const base = borrowedSystemOne();
		assert.ok(base);
		const classifier = withClassifierKeyRotation(base, new Keyring());
		const model = decisionModel("jev-latest");

		const seenKeys: string[] = [];
		const statuses: number[] = [];
		const stubFetch: typeof globalThis.fetch = async (_input, init) => {
			const key = (new Headers(init?.headers).get("authorization") ?? "").replace(/^Bearer\s+/i, "");
			seenKeys.push(key);
			if (key === "key-1") {
				return Response.json(
					{ code: "INSUFFICIENT_BALANCE", error: "Insufficient USD or Diem balance." },
					{ status: 402 },
				);
			}
			return Response.json({
				model: "jev-latest",
				answers: { approved: { type: "noul", noul: 0.91 } },
				usage: { input_tokens: 50, output_tokens: 5 },
			});
		};

		const result = await classifier.classify(model, boolContext, {
			apiKey: "key-1",
			env: { VENICE_API_KEY_2: "key-2" },
			fetch: stubFetch,
			onResponse: (response) => {
				statuses.push(response.status);
			},
		});

		assert.equal(result.stopReason, "stop");
		assert.deepEqual(result.answers, { approved: { type: "bool", probability: 0.91 } });
		assert.deepEqual(seenKeys, ["key-1", "key-2"]);
		assert.deepEqual(statuses, [402, 200]);
	});
});
