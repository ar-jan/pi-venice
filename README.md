# pi-venice

[Venice AI](https://venice.ai) provider for [Pi](https://github.com/earendil-works/pi).

## Install

```bash
pi install git:github.com/ar-jan/pi-venice
```

## Auth

Run `/login venice` to save a primary key in Pi's active credential store, or set `VENICE_API_KEY`.

## Key rotation

A second (or third) key can take over when the current key runs out of credits.

**Stored credential.** Once the active store has a Venice credential, repeat login stops. Pi's
login hook receives no raw credential to append to. To add a fallback, use the first unused
`VENICE_API_KEY_N` starting at 2 across the environment and the active Venice credential's `env`.
For file-backed profiles, edit that profile's `auth.json`. SDK hosts with an injected store can
edit the Venice entry in that store. Keep the primary key (including any `$VAR` or `!command`
reference) and existing fallback numbers as written. To replace the whole credential, run
`/logout venice` and log in again. Logging out also removes its stored fallbacks.

**Environment.** If the primary key comes from `VENICE_API_KEY`, `/login venice` explains which
variable to export instead, because a stored key would override the environment. Add fallbacks with
`VENICE_API_KEY_2`, `VENICE_API_KEY_3`, ... (each may also be a comma- or whitespace-separated list):

```bash
export VENICE_API_KEY=key-1
export VENICE_API_KEY_2=key-2
```

In the active profile's `auth.json`, provider-scoped values take precedence over the process
environment for the same fallback number.

```json
{
  "venice": {
    "type": "api_key",
    "key": "key-1",
    "env": { "VENICE_API_KEY_2": "key-2", "VENICE_API_KEY_3": "key-3" }
  }
}
```

Rotation is transparent and costs no extra requests in normal use: when Venice rejects a request
with HTTP 402 (`INSUFFICIENT_BALANCE`, `API_KEY_USD_SPEND_LIMIT_EXCEEDED`, or
`API_KEY_DIEM_SPEND_LIMIT_EXCEEDED`) before the response starts, the request is retried once on the
next key. The switch is remembered for the process lifetime, so later requests go straight to the
working key. When every key fails, the next request returns to your selected key, or the primary
when none is selected.

Venice does not need Pi's `authHeader` setting, since the key is already sent as
`Authorization: Bearer`. If it is set, the header follows the key through failover and `/venice-key`.
A request whose `Authorization` header (from `headers` in `models.json`) carries any other
credential is sent as configured and never rotated: switching keys could not change it.

**Choosing the active key.** Run `/venice-key` to pick which key a request starts with, or
`/venice-key 2` to select by number. The selector lists each key's source (stored credential,
`VENICE_API_KEY_2`, ...), shows only the last four characters, and marks the current key. The choice
is persisted in `~/.pi/agent/venice-keys.json` as a SHA-256 fingerprint (not the key itself) and is
restored on the next start. Automatic failover still advances to the next key; when all keys fail,
the next request returns to your selected key. Run `/venice-key 1` to go back to the primary.

A 402 that arrives after content has streamed cannot be replayed; the error is passed through, but
the next key is still remembered for later requests. Restart Pi to retry the primary key after
topping it up.

## Update

```bash
pi update git:github.com/ar-jan/pi-venice
```

Then run `/reload` in Pi (or start a new session).

Model details derived from the catalog, such as which thinking levels a model offers, are cached in
`models-store.json`, and pi selects the thinking level against that cache. Interactive pi re-fetches
the Venice catalog in the background on startup (and when `/model` opens) but keeps the model it
already selected, so an update to these details otherwise applies from the next start. Run
`/venice-refresh-models` after updating to apply it right away (see [Model catalog](#model-catalog)).
Until then, a level the cached entry lacks (for example `off`) is clamped to the nearest supported
one.

## Model catalog

`/venice-refresh-models` fetches Venice's catalog now, saves it to `models-store.json`, and reports
the model count and which models were added, removed, or updated. If the current model is a Venice
model whose entry changed, the session switches to the refreshed entry right away. Its thinking
level is kept, except that a configured level the old entry could not offer (for example a default
of `off`) now takes effect. If the refresh fails, the cached models stay in use and the error is
shown.

Pi refreshes catalogs in the background only in interactive and RPC mode: print mode (`pi -p`)
reads the cache, and `pi update --models` does not refresh extension catalogs. The command also
works headlessly, printing its report to stderr:

```bash
pi -p /venice-refresh-models
```

Like `pi update --models`, it fetches even with `--offline` or `PI_OFFLINE` set, since it is an
explicit request.

## Features

- Models are fetched from Venice's live catalog (`GET /models?type=text`), limited to models that
  support tool calling, and cached in `models-store.json`; `/venice-refresh-models` refetches it on
  demand (see [Model catalog](#model-catalog)). Venice's E2EE-capable models are included and run in
  TEE-only mode (see [TEE models](#tee-models)).
- One catalog model is left out because its tool calling fails on every request:
  `e2ee-qwen3-vl-30b-a3b-p` (see [TEE models](#tee-models)).
- Contradictory catalog limits (max output at or above the context window) are clamped so an
  unknown output cap cannot consume the whole window; current Venice models are unaffected.
- Every model name is marked `(private)` or `(anonymized)` after Venice's privacy tier, plus `TEE`
  for models that run in an attested enclave, e.g. `GLM 5.3 Flash (private, TEE)`.
- Reasoning models send the system prompt with the OpenAI `developer` role, which Venice supports.
- Requests carry a per-session `prompt_cache_key` so Venice can route to servers holding the cached
  prefix (omitted when prompt caching is set to `none`, or when a `modelOverrides` entry sets
  `compat.promptCacheKeyMode` to `"disabled"`).
- Claude requests carry their own cache breakpoints as a safeguard, so each tool result stays written
  to the cache once even if Venice's own placement regresses (see
  [Claude prompt caching](#claude-prompt-caching)).
- Venice reports cache writes as `cache_creation_input_tokens`, which Pi's OpenAI-completions adapter
  does not read. The streamed usage is corrected before Pi sees it: written tokens move from `input`
  to `cacheWrite` and the cost is recomputed, so cache reads and Claude's cache-write premium are
  reported accurately.
- The cost Pi shows is what Venice billed, including charges outside the token prices such as web
  search (see [Cost](#cost)).
- Venice's own system prompt is disabled (`venice_parameters.include_venice_system_prompt: false`).
- NUL bytes are stripped from outgoing messages. Venice rejects any message containing U+0000 with
  HTTP 400, so a tool result carrying binary bytes would otherwise break the session from then on:
  both reading a binary file and a program printing decoded bytes end up in the transcript that Pi
  replays with every request.
- The oldest images are replaced with a text placeholder once a request would exceed Venice's
  64-image limit (see [Images](#images)).
- Thinking can be turned off (`off`) where Venice lists a `"none"` effort, and additionally on short
  lists of models verified to honor `venice_parameters.disable_thinking` or an unlisted `"none"`
  (see [Thinking](#thinking)).

## Images

Venice accepts at most 64 images across one request. This limit is not in the Models API or
the docs; it shows up only as an error. (`capabilities.maxImages`, usually 10, behaves as a
per-message limit.) Pi replays the whole transcript with every request, so every image read with
`read` or attached by the user stays in later requests. Once the history holds 65 images, Venice
rejects the request with HTTP 400 `invalid_media: Request exceeds media count limit` (a follow-up
reports `Too many images: 65 (max 64)`), and the session stays stuck because the same history is
sent again.

The extension therefore replaces the oldest images with
`(image omitted: Venice accepts at most 64 images per request)` before sending. It omits them in
blocks of 32 rather than one at a time: the same images stay omitted until the next block is
needed, so the request prefix stays the same and the prompt cache keeps hitting. Dropping only the
overflow would change an early message on every new image and miss the cache for nearly the whole
request each time. Between 33 and 64 of the newest images are sent. The session itself keeps every
image, so the transcript and UI are unchanged.

To keep images out of a long session, have a subagent do screenshot-heavy work and report back in
text, or run `/compact`, which summarizes older messages as text.

## TEE models

Venice's `e2ee-*` models run inside a Trusted Execution Environment (Intel TDX, NVIDIA confidential
computing) with remote attestation. The same model ID serves two modes: E2EE, where the client
encrypts its messages to the enclave and sends Venice's `X-Venice-TEE-*` headers, and TEE-only, an
ordinary request. This extension does not implement the E2EE protocol, so it always uses TEE-only
mode: inference runs in the enclave, but prompts reach the Venice API as plaintext, as with any other
model. The names are therefore tagged `TEE`, never `E2EE`; the IDs keep Venice's `e2ee-` prefix. The
extension does not check the enclave's attestation itself (Venice serves it at
`GET /tee/attestation`); the tag reflects Venice's catalog.

Measured against the live API, two exceptions apply:

- `e2ee-qwen3-vl-30b-a3b-p` is not listed. Its backend rejects every request that carries tools
  (HTTP 400, "\"auto\" tool choice requires --enable-auto-tool-choice ..."), and pi sends its tools
  with every agent request.
- `off` is not offered on TEE models even where Venice lists a `"none"` effort (`e2ee-glm-5-2-p`,
  `e2ee-glm-5-3-p`, `e2ee-kimi-k3-p`): every way of disabling reasoning fails there with HTTP 400
  ("reasoning.max_tokens must be positive"), while each listed effort level works. Selecting `off`
  falls back to the lowest offered level.

TEE models can be slower than their regular counterparts and sometimes run out of capacity (HTTP
429 or 504); pi retries those errors.

## Cost

Venice reports the billed cost of every request at the end of the response stream, split into a USD
and a DIEM part. Pi shows their sum as the request's cost (`usage.cost.total`, marked
`totalFromProvider`), the same way it reports OpenRouter's billed cost. Venice prices DIEM one to one
with USD, so the sum is in the same dollars as the catalog prices.

A token-based estimate misses Venice's other charges. With `enable_web_search` on, Venice bills about
$0.01 per search: a measured search turn cost $0.0105, where its tokens came to $0.0005. The
per-category amounts (input, output, cache read, cache write) remain catalog estimates, so with such
charges they add up to less than the total. A request that ends without Venice's report, for example
when it is aborted, keeps the catalog estimate.

## Claude prompt caching

Claude only caches up to cache breakpoints, which Venice places itself. Since November 2026 its
placement covers the whole prompt, including a `developer` message and the newest tool result, so
each turn is written to the cache once and read back by the next request.

Before that, Venice's breakpoints stopped at the second-to-last user turn (tool results count as
user turns): every tool result was billed as uncached input on the request that sent it, then
written to the cache on the next request. A `developer` message, which pi uses for reasoning models,
got none, so the first request of a session cached nothing.

For `claude-*` models the extension still adds two breakpoints of its own: on the system prompt and
on the last turn, in the form Venice honored for each role (on the message itself for a tool result,
on the last text part for a user message; Venice ignores breakpoints on images). Against Venice's
old placement this cut the cost of an 8-turn tool chain in pi by 31% on Claude Haiku 5.5 and 36% on
Claude Opus 5.5. Against the current placement they change nothing Venice bills; they are kept so
caching stays intact if Venice's placement regresses. The final turn of a conversation is written to
the cache at Claude's write premium even if no request reads it back.

No breakpoints are added when prompt caching is set to `none`. Other models cache automatically and
are unaffected.

## Thinking

`off` is offered when Venice lists a `"none"` reasoning effort, because pi then sends
`reasoning_effort: "none"` and Venice honors it.

Many other reasoning models have no `"none"` option, and there `venice_parameters.disable_thinking`
usually only *hides* the reasoning: Venice keeps generating and billing the tokens, sometimes at the
full default effort. Offering `off` on those models would promise something the request does not
deliver, so it stays unavailable. The exception is a set of models verified against the live API to
really stop reasoning when the toggle is set (completion tokens drop to the answer alone, or Venice
reports zero reasoning tokens). For those, `off` is offered and the request carries
`venice_parameters.disable_thinking: true` instead of a `reasoning_effort`:

`abliteration-abliterated-model-large-flash`, `claude-opus-4-6`, `claude-opus-4-7`,
`claude-opus-4-8`, `gemini-3-5-flash-lite`, `grok-4-20`, `minimax-m3-1-flash`,
`minimax-m3-preview`, `nvidia-nemotron-3-5-super-vl-120b-a12b`, `qwen-3-6-plus`, `qwen-3-7-max`,
`qwen-3-7-plus`, `qwen-3-8-flash`, `qwen3-6-35b-a3b`, `qwen3-235b-a22b-thinking-2507`,
`seed-2-1-turbo`, `zai-org-glm-4.6`, `zai-org-glm-4.7`.

A few models do not list `"none"` but were verified to stop reasoning when sent it anyway, while
the toggle only hides their reasoning. There `off` sends `reasoning_effort: "none"`:
`kimi-k2-7-code`.

Models that reason in the visible answer instead once thinking is off (`claude-opus-4-5`,
`claude-sonnet-4-6`, and `minimax-m25` with `"none"`) do not offer `off`, since it would not shorten
the reply.

Thinking levels otherwise come from the catalog: a model that accepts effort options gets exactly
those levels, while a model without effort control gets one level meaning "model default".

On a verified model, opt out of the toggle for `off` like any other Venice parameter (a layer can
override a key but not remove it, so set an explicit value):

```json
{
  "samplingParamsByThinkingLevel": {
    "off": { "venice_parameters": { "disable_thinking": false } }
  }
}
```

## Venice parameters

Set other [`venice_parameters`](https://docs.venice.ai/api-reference/endpoint/chat/completions) per model in
`~/.pi/agent/models.json`. They are merged with the extension's defaults, so the Venice system prompt
stays off unless you set `include_venice_system_prompt: true`, and thinking stays off at `off` unless
you set `disable_thinking: false`.

Unlike other sampling parameters, `venice_parameters` merges key by key across the model's
`samplingParams`, the thinking level's `samplingParamsByThinkingLevel` entry, and request-level
`samplingParams`, with later layers winning per key. A thinking level can override a key but not
remove it, so set an explicit value such as `"off"`:

```json
{
  "providers": {
    "venice": {
      "modelOverrides": {
        "kimi-k3": {
          "samplingParams": { "venice_parameters": { "enable_web_search": "auto" } },
          "samplingParamsByThinkingLevel": {
            "off": { "venice_parameters": { "enable_web_search": "off" } },
            "high": { "venice_parameters": { "strip_thinking_response": true } }
          }
        }
      }
    }
  }
}
```

## Development

```bash
pi -ne -e .           # one run with only this checkout loaded
pi install .          # or install the working copy in place
npm test              # unit and integration tests (Node 22.6+)
npm run typecheck     # tsc --noEmit
npm run check         # both
```

The integration tests (`test/integration.test.ts`) run the provider's request pipeline through pi's
real OpenAI-completions adapter against a local mock of Venice's chat endpoint; they make no network
requests. The dev dependencies pin the pi packages the tests and typecheck run against; keep them at
the pi release you target (`npm install -D @earendil-works/pi-ai@latest @earendil-works/pi-coding-agent@latest`).
