/**
 * Borrow Pi's built-in TypeSafe System One classifier implementation.
 *
 * Venice documents `POST /api/v1/systemone` as a drop-in alias for TypeSafe's System One
 * endpoint, so reusing Pi's implementation keeps request encoding, retries, error formatting,
 * and usage parsing aligned with Pi without duplicating internal helpers.
 *
 * Pi's extension loader maps `@earendil-works/pi-ai/providers/all` onto the host package,
 * whereas `@earendil-works/pi-ai/api/*` is not mapped and `@earendil-works/pi-ai/compat`
 * only re-exports chat streaming APIs.
 */

import type { ProviderClassifier } from "@earendil-works/pi-ai";
import { builtinProviders } from "@earendil-works/pi-ai/providers/all";

export function borrowedSystemOne(): ProviderClassifier | undefined {
	const classify = builtinProviders().find((p) => p.id === "typesafe")?.classify;
	return classify && { classify: (model, context, options) => classify(model, context, options) };
}
