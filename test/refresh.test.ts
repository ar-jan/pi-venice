import assert from "node:assert/strict";
import { once } from "node:events";
import { describe, it } from "node:test";
import {
	type Api,
	clampThinkingLevel,
	createModels,
	InMemoryCredentialStore,
	InMemoryModelsStore,
	type Model,
	type ModelsRefreshOptions,
	type ModelThinkingLevel,
	type Provider,
} from "@earendil-works/pi-ai";
import registerVenice, { toPiModel } from "../extensions/venice/index.ts";
import { describeCatalogChange, registerProviderWithRefreshCommand } from "../extensions/venice/refresh.ts";

function model(id: string, options: { efforts?: string[]; input?: number } = {}): Model<Api> {
	const result = toPiModel({
		id,
		type: "text",
		model_spec: {
			name: id,
			privacy: "private",
			capabilities: {
				supportsFunctionCalling: true,
				...(options.efforts
					? { supportsReasoning: true, supportsReasoningEffort: true, reasoningEffortOptions: options.efforts }
					: {}),
			},
			pricing: { input: { usd: options.input ?? 1 }, output: { usd: 2 } },
		},
	});
	assert.ok(result);
	return result;
}

interface Notification {
	message: string;
	level: string | undefined;
}

interface Harness {
	run(): Promise<void>;
	notifications: Notification[];
	refreshCalls: unknown[];
	setModelCalls: Model<Api>[];
	session: { model: Model<Api> | undefined; level: ModelThinkingLevel };
}

/**
 * Registers the command against fakes of pi's extension API and model registry. The session mimics
 * pi: selecting a model applies `configuredLevel` and every level is clamped to the selected model.
 */
function harness(options: {
	cached: Model<Api>[];
	fetched?: Model<Api>[];
	current?: Model<Api>;
	level?: ModelThinkingLevel;
	configuredLevel?: ModelThinkingLevel;
	auth?: unknown;
	refresh?: () => Promise<{ aborted: boolean; errors: Map<string, Error> }>;
	publishes?: boolean;
	hasUI?: boolean;
}): Harness {
	let catalog = options.cached;
	let provider: Provider;
	let handler: ((args: string, ctx: unknown) => Promise<void>) | undefined;
	const notifications: Notification[] = [];
	const refreshCalls: unknown[] = [];
	const setModelCalls: Model<Api>[] = [];
	const session = { model: options.current, level: options.level ?? "medium" };
	const configured = options.configuredLevel ?? session.level;

	const pi = {
		registerProvider: (registered: Provider) => {
			provider = registered;
		},
		registerCommand: (name: string, command: { handler: typeof handler }) => {
			assert.equal(name, "venice-refresh-models");
			handler = command.handler;
		},
		getThinkingLevel: () => session.level,
		setThinkingLevel: (level: ModelThinkingLevel) => {
			session.level = session.model ? clampThinkingLevel(session.model, level) : level;
		},
		setModel: async (next: Model<Api>) => {
			setModelCalls.push(next);
			session.model = next;
			session.level = clampThinkingLevel(next, configured);
			return true;
		},
	};
	registerProviderWithRefreshCommand(pi as never, {
		id: "venice",
		async refreshModels(context) {
			const models = options.fetched ?? catalog;
			await context.publish({
				persist: { models, checkedAt: 1 },
				update: () => {
					catalog = models;
				},
			});
		},
	} as Provider);

	const ctx = {
		hasUI: options.hasUI ?? true,
		ui: { notify: (message: string, level?: string) => notifications.push({ message, level }) },
		get model() {
			return session.model;
		},
		modelRegistry: {
			getProviderAuth: async () => ("auth" in options ? options.auth : { auth: { apiKey: "key" } }),
			getModelsOfType: (type: string, provider: string) => {
				assert.deepEqual([type, provider], ["chat", "venice"]);
				return catalog;
			},
			find: (provider: string, id: string) => catalog.find((entry) => entry.provider === provider && entry.id === id),
			refresh: async (refreshOptions: ModelsRefreshOptions) => {
				refreshCalls.push(refreshOptions);
				if (options.refresh) return options.refresh();
				await provider.refreshModels!({
					allowNetwork: refreshOptions.allowNetwork!,
					signal: refreshOptions.signal!,
					publish: async (publication) => {
						if (options.publishes === false) return false;
						publication.update?.();
						return true;
					},
				});
				return { aborted: false, errors: new Map() };
			},
		},
	};

	return {
		run: () => handler!("", ctx),
		notifications,
		refreshCalls,
		setModelCalls,
		session,
	};
}

describe("describeCatalogChange", () => {
	it("reports the count when the cache was empty", () => {
		assert.deepEqual(describeCatalogChange([], [model("a"), model("b")]), ["Venice catalog loaded: 2 models."]);
	});

	it("reports an unchanged catalog", () => {
		assert.deepEqual(describeCatalogChange([model("a")], [model("a")]), ["Venice catalog refreshed: 1 model, no changes."]);
	});

	it("lists added, removed, and updated models", () => {
		const lines = describeCatalogChange(
			[model("keep"), model("gone"), model("priced", { input: 1 })],
			[model("keep"), model("new"), model("priced", { input: 3 })],
		);
		assert.deepEqual(lines, [
			"Venice catalog refreshed: 3 models (1 added, 1 removed, 1 updated).",
			"Added: new",
			"Removed: gone",
			"Updated: priced",
		]);
	});

	it("counts the IDs beyond the listed ones", () => {
		const added = Array.from({ length: 12 }, (_, index) => model(`m${String(index).padStart(2, "0")}`));
		const lines = describeCatalogChange([model("keep")], [model("keep"), ...added]);
		assert.equal(lines[1], "Added: m00, m01, m02, m03, m04, m05, m06, m07, m08, m09, and 2 more");
	});
});

describe("/venice-refresh-models", () => {
	it("warns when Pi discards the catalog during persistence, even if a later command publishes", async (t) => {
		const credentials = new InMemoryCredentialStore();
		await credentials.modify("venice", async () => ({ type: "api_key", key: "test-key" }));
		const modelsStore = new InMemoryModelsStore();
		await modelsStore.write("venice", { models: [model("old")], checkedAt: 1 });
		const models = createModels({ credentials, modelsStore });
		let runCommand!: (args: string, ctx: unknown) => Promise<void>;
		registerVenice({
			on() {},
			registerProvider: (provider: Provider) => models.setProvider(provider),
			registerCommand(name: string, command: { handler: typeof runCommand }) {
				if (name === "venice-refresh-models") runCommand = command.handler;
			},
		} as never);
		await models.refresh({ providers: ["venice"], allowNetwork: false });
		t.mock.method(globalThis, "fetch", async () => Response.json({
			data: [{ id: "new", model_spec: { capabilities: { supportsFunctionCalling: true } } }],
		}));
		const writeStarted = new Promise<void>((resolve) => {
			const waitForAbort: typeof modelsStore.write = async (_id, _entry, options) => {
				resolve();
				await once(options!.signal!, "abort");
			};
			t.mock.method(modelsStore, "write", waitForAbort, { times: 1 });
		});

		const notifications: Notification[] = [];
		const ctx = {
			hasUI: true,
			ui: { notify: (message: string, level: string) => notifications.push({ message, level }) },
			modelRegistry: {
				getProviderAuth: models.getAuth.bind(models),
				getModelsOfType: models.getModelsOfType.bind(models),
				refresh: models.refresh.bind(models),
			},
		};
		await runCommand("", {
			...ctx,
			modelRegistry: {
				...ctx.modelRegistry,
				refresh: async (options: ModelsRefreshOptions) => {
					const first = models.refresh(options);
					await writeStarted;
					await models.refresh({ providers: ["venice"], allowNetwork: false });
					assert.deepEqual(models.getModels("venice").map((entry) => entry.id), ["old"]);
					await runCommand("", ctx);
					return first;
				},
			},
		});

		assert.deepEqual(notifications.map((notice) => notice.level), ["info", "warning"]);
		assert.match(notifications[0]!.message, /Added: new/);
		assert.match(notifications[1]!.message, /interrupted/);
	});

	it("refreshes only Venice, over the network, bypassing freshness checks", async () => {
		const test = harness({ cached: [model("a")], fetched: [model("a"), model("b")] });
		await test.run();
		const [options] = test.refreshCalls as [{ providers: string[]; allowNetwork: boolean; force: boolean; signal: AbortSignal }];
		assert.deepEqual(options.providers, ["venice"]);
		assert.equal(options.allowNetwork, true);
		assert.equal(options.force, true);
		assert.ok(options.signal instanceof AbortSignal);
		assert.deepEqual(test.notifications, [
			{ message: "Venice catalog refreshed: 2 models (1 added).\nAdded: b", level: "info" },
		]);
	});

	it("does not refresh without a configured key", async () => {
		const test = harness({ cached: [], auth: undefined });
		await test.run();
		assert.equal(test.refreshCalls.length, 0);
		assert.equal(test.notifications[0]?.level, "warning");
		assert.match(test.notifications[0]!.message, /No Venice API key configured/);
	});

	it("reports a failed refresh and keeps the current model", async () => {
		const current = model("a");
		const test = harness({
			cached: [current],
			current,
			refresh: async () => ({ aborted: false, errors: new Map([["venice", new Error("Venice model list request failed: 401")]]) }),
		});
		await test.run();
		assert.deepEqual(test.notifications, [
			{
				message: "Could not refresh the Venice catalog: Venice model list request failed: 401. Keeping the cached models.",
				level: "error",
			},
		]);
		assert.equal(test.setModelCalls.length, 0);
	});

	it("reports a timeout", async () => {
		const test = harness({ cached: [], refresh: async () => ({ aborted: true, errors: new Map() }) });
		await test.run();
		assert.match(test.notifications[0]!.message, /timed out/);
	});

	it("says so when another refresh superseded this one", async () => {
		const test = harness({ cached: [model("a")], current: model("a", { input: 5 }), publishes: false });
		await test.run();
		assert.equal(test.notifications[0]?.level, "warning");
		assert.match(test.notifications[0]!.message, /Run \/venice-refresh-models again/);
		assert.equal(test.setModelCalls.length, 0);
	});

	it("applies a configured level the cached entry could not offer", async () => {
		// The cached entry lacked "off", so the configured "off" was clamped at startup.
		const cached = model("a", { efforts: ["low", "medium", "high"] });
		const fetched = model("a", { efforts: ["none", "low", "medium", "high"] });
		const startLevel = clampThinkingLevel(cached, "off");
		assert.notEqual(startLevel, "off");
		const test = harness({ cached: [cached], fetched: [fetched], current: cached, level: startLevel, configuredLevel: "off" });
		await test.run();
		assert.deepEqual(test.setModelCalls, [fetched]);
		assert.equal(test.session.level, "off");
		assert.deepEqual(test.notifications[0]?.message.split("\n"), [
			"Venice catalog refreshed: 1 model (1 updated).",
			"Updated: a",
			"Applied to the current model a.",
			"Thinking levels: low, medium, high → off, low, medium, high.",
			`Thinking level: ${startLevel} → off.`,
		]);
	});

	it("keeps a level chosen during the session", async () => {
		const cached = model("a", { efforts: ["low", "medium", "high"] });
		const fetched = model("a", { efforts: ["none", "low", "medium", "high"] });
		const test = harness({ cached: [cached], fetched: [fetched], current: cached, level: "high", configuredLevel: "off" });
		await test.run();
		assert.equal(test.session.model, fetched);
		assert.equal(test.session.level, "high");
		assert.doesNotMatch(test.notifications[0]!.message, /Thinking level:/);
	});

	it("leaves an unchanged current model alone", async () => {
		const current = model("a");
		const test = harness({ cached: [current], fetched: [model("a")], current });
		await test.run();
		assert.equal(test.setModelCalls.length, 0);
		assert.equal(test.notifications[0]?.message, "Venice catalog refreshed: 1 model, no changes.");
	});

	it("leaves a model from another provider alone", async () => {
		const other = { ...model("a", { input: 5 }), provider: "openai" };
		const test = harness({ cached: [model("a")], fetched: [model("a", { input: 9 })], current: other });
		await test.run();
		assert.equal(test.setModelCalls.length, 0);
	});

	it("warns when the current model left the catalog", async () => {
		const current = model("gone");
		const test = harness({ cached: [current, model("a")], fetched: [model("a")], current });
		await test.run();
		assert.equal(test.setModelCalls.length, 0);
		assert.match(test.notifications[0]!.message, /current model gone is no longer offered/);
	});

	it("writes to stderr without a UI", async (t) => {
		const errors: unknown[] = [];
		t.mock.method(console, "error", (message: unknown) => errors.push(message));
		const test = harness({ cached: [model("a")], hasUI: false });
		await test.run();
		assert.deepEqual(errors, ["Venice catalog refreshed: 1 model, no changes."]);
		assert.equal(test.notifications.length, 0);
	});
});
