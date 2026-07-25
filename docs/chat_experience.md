# Chat Experience

Phase 5 adds a more complete local chat experience on top of the shared runtime in `src/chat_runtime.py`.

## What changed

* streaming replies in the terminal CLI
* persona presets stored in `configs/personas.json`
* session save and resume support through `session_file`
* a lightweight local browser UI in `scripts/web_chat.py`
* one shared generation path for CLI and browser usage
* hidden reasoning separated from the visible `<|final|>` answer
* a checkpointed long-horizon task runner

## CLI workflow

Start the terminal chat:

```bash
python scripts/chat.py --config configs/chat.yaml
```

Useful options:

* `--list-personas` lists available personas and exits
* `--persona coach` starts with a named persona
* `--session-file logs/my_chat.json` overrides the default session path
* `--json-mode` starts the session in JSON mode
* `--no-stream` disables incremental token printing

Interactive commands:

* `:help`
* `:clear`
* `:history`
* `:save`
* `:persona NAME`
* `:json on|off`
* `:system TEXT`
* `:session`

## Web workflow

Start the local web UI:

```bash
python scripts/web_chat.py --config configs/chat.yaml
```

By default the server runs at `http://127.0.0.1:8000`.

Optional flags:

* `--host 0.0.0.0`
* `--port 8080`
* `--persona json_bot`
* `--session-file logs/browser_chat.json`

The browser UI supports:

* persona switching
* system prompt editing
* JSON mode toggling
* history clearing
* session persistence with the same JSON format as the CLI

## Session format

Saved sessions are plain JSON objects with:

* `persona`
* `system_prompt`
* `json_mode`
* `updated_at`
* `history`

Each history entry stores `user` and `assistant` text so a session can be reopened by either chat surface.

## Long-horizon tasks

Create a task and run up to eight model steps in this process:

```powershell
python scripts/run_agent_task.py --state logs/research_task.json --objective "Complete the requested multi-step task and report verified results." --steps-per-run 8
```

Resume the same state without repeating the objective:

```powershell
python scripts/run_agent_task.py --state logs/research_task.json --steps-per-run 8
```

The default total budget is 64 model steps. Use `--max-steps` to extend it and
`--max-wall-time-seconds` to checkpoint and pause on a wall-clock boundary.
State is written atomically after every model/tool round.
