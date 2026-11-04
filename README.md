# pi-venice

[Venice AI](https://venice.ai) provider for [Pi](https://github.com/earendil-works/pi).

## Install

```bash
pi install git:github.com/ar-jan/pi-venice
```

## Auth

`/login venice` (stored in `auth.json`), or via the `VENICE_API_KEY` environment variable.

## Key rotation

A second (or third) key can take over when the current key runs out of credits.

**Stored credential (recommended).** Run `/login venice` once to store the primary key. Running it
again prompts for another key and stores it in `auth.json` under the credential's `env` as
`VENICE_API_KEY_2`; further runs use `VENICE_API_KEY_3`, and so on. The stored primary stays first,
and a primary stored as a `$VAR` or `!command` reference is kept as written. To replace the primary
key, run `/logout venice` and log in again (this also removes the stored fallback keys). If
`auth.json` holds a Venice key that cannot be read, login stops instead of replacing it.

**Environment.** If the primary key comes from `VENICE_API_KEY`, `/login venice` explains which
variable to export instead, because a stored key would override the environment. Add fallbacks with
`VENICE_API_KEY_2`, `VENICE_API_KEY_3`, ... (each may also be a comma- or whitespace-separated list):

```bash
export VENICE_API_KEY=key-1
export VENICE_API_KEY_2=key-2
```

Keys can also be edited directly in `auth.json` (provider-scoped values take precedence over the
process environment):

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

## Features

- Models are fetched from Venice's live catalog (`GET /models?type=text`), limited to models that
support tool calling and do not require end-to-end encryption (E2EE), and cached in
`models-store.json`.
- Contradictory catalog limits (max output at or above the context window) are clamped so an
  unknown output cap cannot consume the whole window; current Venice models are unaffected.
- Every model name is marked `(private)` or `(anonymized)` after Venice's privacy tier.
- Reasoning models send the system prompt with the OpenAI `developer` role, which Venice supports.
- Requests carry a per-session `prompt_cache_key` so Venice can route to servers holding the cached
  prefix (omitted when prompt caching is set to `none`).
- Claude requests carry their own cache breakpoints, so each tool result is written to the cache once
  instead of being billed as uncached input first (see [Claude prompt caching](#claude-prompt-caching)).
- Venice reports cache writes as `cache_creation_input_tokens`, which Pi's OpenAI-completions adapter
  does not read. The streamed usage is corrected before Pi sees it: written tokens move from `input`
  to `cacheWrite` and the cost is recomputed, so cache reads and Claude's cache-write premium are
  reported accurately.
- Venice's own system prompt is disabled (`venice_parameters.include_venice_system_prompt: false`).
- NUL bytes are stripped from outgoing messages. Venice rejects any message containing U+0000 with
  HTTP 400, so a tool result carrying binary bytes would otherwise break the session from then on:
  both reading a binary file and a program printing decoded bytes end up in the transcript that Pi
  replays with every request.
- Thinking can be turned off (`off`) where Venice lists a `"none"` effort, and additionally on a
  short list of models verified to honor `venice_parameters.disable_thinking` (see [Thinking](#thinking)).

## Claude prompt caching

Claude only caches up to explicit breakpoints. Venice adds its own on a `system` message and on the
second-to-last user turn (tool results count as user turns), which leaves the newest turn out: every
tool result is billed as uncached input on the request that sends it, then written to the cache on
the next request. Venice also adds none for a `developer` message, which pi uses for reasoning
models, so on its own the first request of a session caches nothing.

For `claude-*` models the extension therefore adds two breakpoints of its own: on the system prompt
and on the last turn, in the form Venice honors for each role (on the message itself for a tool
result, on the last text part for a user message; Venice ignores breakpoints on images). Each turn is
then written once and read back by the next request. In an 8-turn tool chain measured in pi this cut
the cost by 31% on Claude Haiku 5.5 and 36% on Claude Opus 5.5. The final turn of a conversation is
written to the cache at Claude's write premium even if no request reads it back.

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
`gemini-3-5-flash-lite`, `grok-4-20`, `minimax-m3-preview`, `qwen-3-6-plus`, `qwen-3-7-max`,
`qwen-3-7-plus`, `qwen-3-8-flash`, `qwen3-6-35b-a3b`, `qwen3-235b-a22b-thinking-2507`,
`seed-2-1-turbo`, `zai-org-glm-4.6`, `zai-org-glm-4.7`.

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
