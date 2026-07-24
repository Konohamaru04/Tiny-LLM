# Dataset contract

## Pretraining documents

Pretraining accepts UTF-8 Markdown anywhere under `data/raw/`. Documents are
normalized, content-hashed, and deduplicated before a deterministic
document-level train/validation split. Identical files in multiple directories
are therefore included once.

The current local snapshot contains:

- 5,000 pinned FineWeb-Edu documents in `data/raw/public_fineweb/`
- 297 additional unique Markdown documents, mirrored under `data/raw/` and
  `data/raw/public/`
- 5,297 unique pretraining documents after skipping 297 duplicate copies

The tokenizer metadata records discovered, unique, and skipped-duplicate
counts. The 297 additional documents were supplied locally; confirm their
provenance and usage rights before redistributing trained artifacts.

## SFT JSONL

One valid JSON object is required per line. The preferred schema follows
common chat-completion conventions:

```json
{
  "id": "example-1",
  "source": "dataset-name",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 19 * 23?"},
    {
      "role": "assistant",
      "reasoning_content": "I should calculate this exactly.",
      "content": "",
      "tool_calls": [
        {
          "id": "call-1",
          "type": "function",
          "function": {
            "name": "calculator",
            "arguments": "{\"expression\":\"19 * 23\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call-1",
      "name": "calculator",
      "content": "{\"ok\":true,\"result\":{\"value\":437}}"
    },
    {"role": "assistant", "content": "19 * 23 is 437."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Evaluate arithmetic.",
        "parameters": {"type": "object"}
      }
    }
  ]
}
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
Assistant messages may contain `reasoning_content`, `content`, and
`tool_calls`. Tool messages should include `tool_call_id`. Serialization wraps
assistant-visible content in `<|final|>`; this token is supervised separately
from `<|think|>` reasoning and tool calls.

The legacy `system` / `user` / `assistant` row format remains accepted for
small custom datasets.

## Quality rules

- Keep train and validation records disjoint.
- Deduplicate exact normalized examples.
- Preserve tool schemas, call IDs, and matching tool responses.
- Reject examples without an assistant training target.
- Use dynamic per-batch padding; labels on padding and non-assistant context
  must remain `-100`.
- Keep private secrets and personally identifying information out of data.
- Treat reasoning traces as training data, not guaranteed factual evidence.
- Review each source's license, terms, and upstream model restrictions.
