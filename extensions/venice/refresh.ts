/**
 * `/venice-refresh-models`: refetch Venice's catalog on demand.
 *
 * Pi refreshes extension catalogs in the background (interactive startup, `/model`, `/login`),
 * reports nothing about the result, and keeps the session's selected model object until the model
 * is selected again, so updated details such as its thinking levels apply only from the next start.
 * The command runs the same provider-scoped refresh as `pi update --models` (network allowed and
 * freshness bypassed, since the user asked for it explicitly), reports what changed, and applies the
 * refreshed entry to the current model. Pi persists the result in models-store.json, so print mode
 * (`pi -p /venice-refresh-models`) also works as a headless refresh.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import {
	type Api,
	clampThinkingLevel,
	getSupportedThinkingLevels,
	type Model,
	type Provider,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";

/** Covers key resolution plus the catalog request's own 15 s timeout. */
const REFRESH_TIMEOUT_MS = 20_000;
/** Model IDs listed per change kind before the rest are counted. */
const MAX_LISTED_IDS = 10;

type NotifyLevel = "info" | "warning" | "error";

/** Notifications are no-ops without a UI (print and JSON modes); write those to stderr. */
function report(ctx: ExtensionCommandContext, message: string, level: NotifyLevel): void {
	if (ctx.hasUI) ctx.ui.notify(message, level);
	else console.error(message);
}

function sameModel(a: Model<Api>, b: Model<Api>): boolean {
	return JSON.stringify(a) === JSON.stringify(b);
}

function listIds(label: string, ids: string[]): string | undefined {
	if (ids.length === 0) return undefined;
	const shown = ids.slice(0, MAX_LISTED_IDS).join(", ");
	const rest = ids.length - MAX_LISTED_IDS;
	return `${label}: ${shown}${rest > 0 ? `, and ${rest} more` : ""}`;
}

/** A summary of how the provider's catalog changed, one line per kind of change. */
export function describeCatalogChange(before: readonly Model<Api>[], after: readonly Model<Api>[]): string[] {
	const count = `${after.length} model${after.length === 1 ? "" : "s"}`;
	if (before.length === 0) return [`Venice catalog loaded: ${count}.`];

	const previous = new Map(before.map((model) => [model.id, model]));
	const current = new Set(after.map((model) => model.id));
	const added: string[] = [];
	const updated: string[] = [];
	for (const model of after) {
		const old = previous.get(model.id);
		if (!old) added.push(model.id);
		else if (!sameModel(old, model)) updated.push(model.id);
	}
	const removed = before.map((model) => model.id).filter((id) => !current.has(id));
	const byId = (a: string, b: string) => a.localeCompare(b);
	added.sort(byId);
	removed.sort(byId);
	updated.sort(byId);

	const changes = [
		added.length ? `${added.length} added` : "",
		removed.length ? `${removed.length} removed` : "",
		updated.length ? `${updated.length} updated` : "",
	].filter(Boolean);
	if (changes.length === 0) return [`Venice catalog refreshed: ${count}, no changes.`];
	return [
		`Venice catalog refreshed: ${count} (${changes.join(", ")}).`,
		...[listIds("Added", added), listIds("Removed", removed), listIds("Updated", updated)].filter(
			(line): line is string => line !== undefined,
		),
	];
}

/**
 * Replace the session's model object with its refreshed catalog entry. Pi keeps the object it
 * selected, so without this the refreshed details (thinking levels, limits, prices) would apply only
 * after selecting the model again or restarting.
 */
async function applyToCurrentModel(
	pi: ExtensionAPI,
	ctx: ExtensionCommandContext,
	providerId: string,
): Promise<string[]> {
	const current = ctx.model;
	if (current?.provider !== providerId) return [];
	const refreshed = ctx.modelRegistry.find(providerId, current.id);
	if (!refreshed) {
		return [`The current model ${current.id} is no longer offered by Venice; choose another with /model.`];
	}
	if (sameModel(current, refreshed)) return [];

	const previousLevel = pi.getThinkingLevel();
	if (!(await pi.setModel(refreshed))) return [];
	// Selecting a model applies its configured thinking level (per-model setting, else the default),
	// clamped to the refreshed levels. If the session's level is what that configured level clamped to
	// on the old entry, it came from the configuration and the refreshed one wins: this is how a
	// default of `off` takes effect once the refreshed entry offers it. Otherwise the user chose the
	// level during the session, and it is kept (clamped to what the refreshed entry offers).
	if (clampThinkingLevel(current, pi.getThinkingLevel()) !== previousLevel) {
		pi.setThinkingLevel(previousLevel);
	}

	const lines = [`Applied to the current model ${current.id}.`];
	const oldLevels = getSupportedThinkingLevels(current).join(", ");
	const newLevels = getSupportedThinkingLevels(refreshed).join(", ");
	if (oldLevels !== newLevels) lines.push(`Thinking levels: ${oldLevels} → ${newLevels}.`);
	const level = pi.getThinkingLevel();
	if (level !== previousLevel) lines.push(`Thinking level: ${previousLevel} → ${level}.`);
	return lines;
}

/** Register the dynamic provider and report a command's success once Pi accepts its fetched catalog. */
export function registerProviderWithRefreshCommand(pi: ExtensionAPI, provider: Provider): void {
	const providerId = provider.id;
	const activeCommand = new AsyncLocalStorage<{ published: boolean }>();
	pi.registerProvider({
		...provider,
		refreshModels(context) {
			const command = activeCommand.getStore();
			if (!command || !context.allowNetwork) return provider.refreshModels!(context);
			return provider.refreshModels!({
				...context,
				publish: async (publication) => {
					const accepted = await context.publish(publication);
					// Pi also publishes the cached catalog during refresh. Only the fetched catalog has a store entry.
					if (accepted && publication.persist) command.published = true;
					return accepted;
				},
			});
		},
	});

	pi.registerCommand("venice-refresh-models", {
		description: "Fetch the Venice model catalog now and apply it to the current model",
		handler: async (_args, ctx) => {
			// Pi only fetches a catalog for a configured provider and otherwise returns no error.
			const auth = await ctx.modelRegistry.getProviderAuth(providerId).catch(() => null);
			if (auth === undefined) {
				report(ctx, "No Venice API key configured. Run /login venice or set VENICE_API_KEY.", "warning");
				return;
			}

			const before = ctx.modelRegistry.getModelsOfType("chat", providerId);
			const command = { published: false };
			let result: Awaited<ReturnType<ExtensionCommandContext["modelRegistry"]["refresh"]>>;
			try {
				result = await activeCommand.run(command, () =>
					ctx.modelRegistry.refresh({
						providers: [providerId],
						allowNetwork: true,
						force: true,
						signal: AbortSignal.timeout(REFRESH_TIMEOUT_MS),
					}),
				);
			} catch (error) {
				result = { aborted: false, errors: new Map([[providerId, error as Error]]) };
			}

			const error = result.errors.get(providerId);
			if (error) {
				const message = error instanceof Error ? error.message : String(error);
				report(ctx, `Could not refresh the Venice catalog: ${message}. Keeping the cached models.`, "error");
				return;
			}
			if (result.aborted) {
				report(ctx, "The Venice catalog refresh timed out. Keeping the cached models.", "error");
				return;
			}
			if (!command.published) {
				report(ctx, "Another model refresh interrupted this one. Run /venice-refresh-models again.", "warning");
				return;
			}

			const lines = describeCatalogChange(before, ctx.modelRegistry.getModelsOfType("chat", providerId));
			lines.push(...(await applyToCurrentModel(pi, ctx, providerId)));
			report(ctx, lines.join("\n"), "info");
		},
	});
}
