# pi-venice

[Venice AI](https://venice.ai) provider for [Pi](https://github.com/earendil-works/pi).

- **Auth:** `/login venice` (stored in `auth.json`) or the `VENICE_API_KEY` environment variable.
- **Models:** fetched from Venice's live catalog (`GET /models?type=text`), limited to models that
  support tool calling, and cached in `models-store.json`.
- Every model name is marked `(private)` or `(anonymized)` after Venice's privacy tier.
- Requests carry a per-session `prompt_cache_key` so Venice can route to servers holding the cached
  prefix (omitted when prompt caching is set to `none`).
- Venice's own system prompt is disabled (`venice_parameters.include_venice_system_prompt: false`).

## Venice parameters

Set other [`venice_parameters`](https://docs.venice.ai/api-reference/endpoint/chat/completions) per model in
`~/.pi/agent/models.json`. They are merged with the default above, so the Venice system prompt stays
off unless you set `include_venice_system_prompt: true`:

```json
{
  "providers": {
    "venice": {
      "modelOverrides": {
        "kimi-k3": {
          "samplingParams": { "venice_parameters": { "enable_web_search": "auto" } }
        }
      }
    }
  }
}
```

## Install

```bash
pi install git:github.com/ar-jan/pi-venice
```

# Update

```bash
pi update git:github.com/ar-jan/pi-venice
```

Then run `/reload` in Pi (or start a new session).

## Development

```bash
pi -ne -e .           # one run with only this checkout loaded
pi install .          # or install the working copy in place
npm test              # unit tests (Node 22.6+)
```
