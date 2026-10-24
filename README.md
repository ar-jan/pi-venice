# pi-venice

[Venice AI](https://venice.ai) provider for [Pi](https://github.com/earendil-works/pi).

## Install

```bash
pi install git:github.com/ar-jan/pi-venice
```

## Auth

`/login venice` (stored in `auth.json`), or via the `VENICE_API_KEY` environment variable.

## Update

```bash
pi update git:github.com/ar-jan/pi-venice
```

Then run `/reload` in Pi (or start a new session).

## Features

- Models are fetched from Venice's live catalog (`GET /models?type=text`), limited to models that
support tool calling and do not require end-to-end encryption (E2EE), and cached in
`models-store.json`.
- Every model name is marked `(private)` or `(anonymized)` after Venice's privacy tier.
- Requests carry a per-session `prompt_cache_key` so Venice can route to servers holding the cached
  prefix (omitted when prompt caching is set to `none`).
- Venice's own system prompt is disabled (`venice_parameters.include_venice_system_prompt: false`).
- Thinking can be turned off (`off`) where Venice lists a `"none"` effort, and additionally on a
  short list of models verified to honor `venice_parameters.disable_thinking` (see [Thinking](#thinking)).

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
npm test              # unit tests (Node 22.6+)
```
