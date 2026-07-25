from __future__ import annotations

import argparse
import json
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat_runtime import (
    generate_chat_response,
    load_chat_runtime,
    load_personas,
    load_session,
    resolve_persona,
    run_agent_turn,
    save_session,
)
from src.config import load_chat_config
from src.tools import build_default_tool_registry


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tiny LLM Web Chat</title>
  <style>
    :root {
      --bg: #0d1117;
      --surface: #161b22;
      --surface-raised: #1c2128;
      --surface-input: #0d1117;
      --border: rgba(255, 255, 255, 0.08);
      --border-strong: rgba(255, 255, 255, 0.14);
      --text: #e6edf3;
      --muted: #7d8590;
      --accent: #2dd4bf;
      --accent-dim: rgba(45, 212, 191, 0.12);
      --accent-border: rgba(45, 212, 191, 0.22);
      --user-bg: rgba(45, 212, 191, 0.07);
      --user-border: rgba(45, 212, 191, 0.16);
      --assistant-bg: rgba(255, 255, 255, 0.04);
      --btn-primary: #2dd4bf;
      --btn-primary-text: #0d1117;
      --shadow: 0 0 0 1px var(--border), 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    html, body {
      height: 100%;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, "Segoe UI", system-ui, sans-serif;
      font-size: 14px;
      line-height: 1.6;
    }

    .shell {
      display: grid;
      grid-template-columns: 300px 1fr;
      height: 100vh;
      overflow: hidden;
    }

    /* ── Sidebar ── */
    .sidebar {
      background: var(--surface);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .sidebar-scroll {
      flex: 1;
      overflow-y: auto;
      padding: 20px 16px;
      scrollbar-width: thin;
      scrollbar-color: var(--border-strong) transparent;
    }

    .sidebar-scroll::-webkit-scrollbar { width: 4px; }
    .sidebar-scroll::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 2px; }

    .brand {
      margin-bottom: 20px;
    }

    .eyebrow {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--accent);
      margin-bottom: 4px;
    }

    h1 {
      font-size: 1.5rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--text);
    }

    .lead {
      margin-top: 6px;
      font-size: 0.82rem;
      color: var(--muted);
      line-height: 1.5;
    }

    .meta {
      margin-top: 14px;
      padding: 10px 12px;
      border-radius: 10px;
      background: var(--surface-raised);
      border: 1px solid var(--border);
      display: grid;
      gap: 4px;
      font-size: 0.8rem;
    }

    .meta strong { color: var(--text); }
    .meta .hint { color: var(--muted); }

    .section-label {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
      margin: 18px 0 8px;
    }

    label {
      display: block;
      font-size: 0.8rem;
      color: var(--muted);
      margin-bottom: 5px;
    }

    select, textarea, input[type="text"] {
      width: 100%;
      padding: 8px 10px;
      background: var(--surface-input);
      color: var(--text);
      border: 1px solid var(--border-strong);
      border-radius: 8px;
      font: inherit;
      font-size: 0.85rem;
      outline: none;
      transition: border-color 150ms;
    }

    select:focus, textarea:focus, input[type="text"]:focus {
      border-color: var(--accent);
    }

    select option { background: var(--surface); }

    textarea { resize: vertical; }

    .hint { color: var(--muted); font-size: 0.8rem; margin-top: 5px; line-height: 1.4; }

    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid var(--border);
    }

    button {
      width: 100%;
      padding: 8px 12px;
      border-radius: 8px;
      border: 1px solid transparent;
      font: inherit;
      font-size: 0.85rem;
      font-weight: 500;
      cursor: pointer;
      transition: opacity 120ms, transform 120ms;
    }

    button:hover:not(:disabled) { opacity: 0.85; transform: translateY(-1px); }
    button:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

    button.primary {
      background: var(--btn-primary);
      color: var(--btn-primary-text);
    }

    button.secondary {
      background: transparent;
      color: var(--muted);
      border-color: var(--border-strong);
    }

    button.secondary:hover:not(:disabled) { color: var(--text); border-color: var(--border-strong); }

    /* ── Main ── */
    .main {
      display: flex;
      flex-direction: column;
      overflow: hidden;
      background: var(--bg);
    }

    .transcript {
      flex: 1;
      overflow-y: auto;
      padding: 16px 20px;
      scrollbar-width: thin;
      scrollbar-color: var(--border-strong) transparent;
    }

    .transcript::-webkit-scrollbar { width: 4px; }
    .transcript::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 2px; }

    .message {
      margin-bottom: 12px;
      padding: 12px 14px;
      border-radius: 12px;
      white-space: pre-wrap;
      line-height: 1.6;
      font-size: 0.9rem;
    }

    .message.user {
      background: var(--user-bg);
      border: 1px solid var(--user-border);
      margin-left: 10%;
    }

    .message.assistant {
      background: var(--assistant-bg);
      border: 1px solid var(--border);
      margin-right: 6%;
    }

    .role {
      display: inline-block;
      margin-bottom: 6px;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--accent);
      font-weight: 600;
    }

    .message.assistant .role { color: var(--muted); }

    .empty {
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 8px;
      text-align: center;
      color: var(--muted);
      padding: 40px;
    }

    .empty-icon {
      font-size: 2rem;
      opacity: 0.3;
      margin-bottom: 8px;
    }

    .empty strong { color: var(--text); font-size: 0.95rem; }
    .empty span { font-size: 0.82rem; }

    /* ── Composer ── */
    .composer {
      border-top: 1px solid var(--border);
      padding: 12px 20px 14px;
      background: var(--surface);
      display: flex;
      flex-direction: column;
      gap: 8px;
      flex-shrink: 0;
    }

    .composer textarea {
      min-height: 72px;
      max-height: 180px;
    }

    .composer-footer {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 8px;
      background: var(--accent-dim);
      border: 1px solid var(--accent-border);
      color: var(--text);
      font-size: 0.8rem;
      cursor: pointer;
      white-space: nowrap;
      flex-shrink: 0;
    }

    .toggle input[type="checkbox"] {
      width: 14px;
      height: 14px;
      accent-color: var(--accent);
      margin: 0;
    }

    .send-btn {
      flex: 1;
      padding: 8px 16px;
      background: var(--btn-primary);
      color: var(--btn-primary-text);
      font-weight: 600;
    }

    .status {
      font-size: 0.78rem;
      color: var(--muted);
      min-height: 1.2rem;
    }

    @media (max-width: 860px) {
      .shell { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
      .sidebar {
        border-right: none;
        border-bottom: 1px solid var(--border);
        max-height: 40vh;
      }
      .message.user { margin-left: 0; }
      .message.assistant { margin-right: 0; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="sidebar-scroll">
        <div class="brand">
          <div class="eyebrow">Local Runtime</div>
          <h1>Tiny LLM</h1>
          <p class="lead">Browser shell for the local checkpoint with persona and session support.</p>
        </div>

        <div class="meta">
          <strong id="device">device: --</strong>
          <span id="session-path" class="hint"></span>
        </div>

        <div class="section-label">Persona</div>
        <label for="persona-select">Active preset</label>
        <select id="persona-select"></select>
        <p id="persona-description" class="hint"></p>

        <div class="section-label">System Prompt</div>
        <label for="system-prompt">Prompt used for new turns</label>
        <textarea id="system-prompt" rows="5"></textarea>

        <div class="actions">
          <button id="save-settings" class="primary" type="button">Save</button>
          <button id="clear-history" class="secondary" type="button">Clear chat</button>
        </div>
      </div>
    </aside>

    <main class="main">
      <div id="transcript" class="transcript"></div>

      <div class="composer">
        <textarea id="message-box" placeholder="Ask something… (Ctrl+Enter to send)"></textarea>
        <div class="composer-footer">
          <label class="toggle" for="json-mode">
            <input id="json-mode" type="checkbox">
            JSON mode
          </label>
          <button id="send-button" class="primary send-btn" type="button">Send</button>
        </div>
        <div id="status" class="status"></div>
      </div>
    </main>
  </div>

  <script>
    const transcriptEl = document.getElementById("transcript");
    const personaSelect = document.getElementById("persona-select");
    const personaDescription = document.getElementById("persona-description");
    const systemPromptEl = document.getElementById("system-prompt");
    const messageBoxEl = document.getElementById("message-box");
    const jsonModeEl = document.getElementById("json-mode");
    const statusEl = document.getElementById("status");
    const deviceEl = document.getElementById("device");
    const sessionPathEl = document.getElementById("session-path");
    const sendButtonEl = document.getElementById("send-button");
    const saveSettingsEl = document.getElementById("save-settings");
    const clearHistoryEl = document.getElementById("clear-history");

    let appState = null;

    function escapeHtml(text) {
      const div = document.createElement("div");
      div.textContent = text;
      return div.innerHTML;
    }

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function renderTranscript(history) {
      if (!history.length) {
        transcriptEl.innerHTML = '<div class="empty"><div class="empty-icon">&#x1F4AC;</div><strong>No conversation yet.</strong><span>Start with a question below or switch personas first.</span></div>';
        return;
      }

      transcriptEl.innerHTML = history.map((turn) => `
        <div class="message user">
          <div class="role">User</div>
          <div>${escapeHtml(turn.user)}</div>
        </div>
        <div class="message assistant">
          <div class="role">Assistant</div>
          <div>${escapeHtml(turn.assistant)}</div>
        </div>
      `).join("");
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
    }

    function renderPersonas(personas, activePersona) {
      personaSelect.innerHTML = personas.map((persona) => {
        const selected = persona.name === activePersona ? "selected" : "";
        return `<option value="${escapeHtml(persona.name)}" ${selected}>${escapeHtml(persona.name)}</option>`;
      }).join("");

      const selectedPersona = personas.find((persona) => persona.name === activePersona);
      personaDescription.textContent = selectedPersona ? selectedPersona.description || "" : "";
    }

    function applyState(state) {
      appState = state;
      deviceEl.textContent = `device: ${state.device}`;
      sessionPathEl.textContent = state.session_file ? `session: ${state.session_file}` : "session: disabled";
      renderPersonas(state.personas, state.persona);
      systemPromptEl.value = state.system_prompt;
      jsonModeEl.checked = state.json_mode;
      renderTranscript(state.history);
    }

    async function loadState() {
      const response = await fetch("/api/state");
      if (!response.ok) {
        throw new Error("Failed to load state");
      }
      const payload = await response.json();
      applyState(payload);
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({ error: "Request failed." }));
        throw new Error(error.error || "Request failed");
      }
      return response.json();
    }

    async function sendMessage() {
      const message = messageBoxEl.value.trim();
      if (!message) {
        setStatus("Write a message first.");
        return;
      }

      sendButtonEl.disabled = true;
      setStatus("Generating response...");
      try {
        const payload = await postJson("/api/chat", {
          message,
          persona: personaSelect.value,
          system_prompt: systemPromptEl.value.trim(),
          json_mode: jsonModeEl.checked,
        });
        applyState(payload.state);
        messageBoxEl.value = "";
        setStatus("Response ready.");
      } catch (error) {
        setStatus(error.message);
      } finally {
        sendButtonEl.disabled = false;
        messageBoxEl.focus();
      }
    }

    async function saveSettings() {
      saveSettingsEl.disabled = true;
      setStatus("Saving settings...");
      try {
        const payload = await postJson("/api/settings", {
          persona: personaSelect.value,
          system_prompt: systemPromptEl.value.trim(),
          json_mode: jsonModeEl.checked,
        });
        applyState(payload.state);
        setStatus("Settings saved.");
      } catch (error) {
        setStatus(error.message);
      } finally {
        saveSettingsEl.disabled = false;
      }
    }

    async function clearHistory() {
      clearHistoryEl.disabled = true;
      setStatus("Clearing chat...");
      try {
        const payload = await postJson("/api/clear", {});
        applyState(payload.state);
        setStatus("Conversation cleared.");
      } catch (error) {
        setStatus(error.message);
      } finally {
        clearHistoryEl.disabled = false;
      }
    }

    personaSelect.addEventListener("change", () => {
      const selectedPersona = appState.personas.find((persona) => persona.name === personaSelect.value);
      personaDescription.textContent = selectedPersona ? selectedPersona.description || "" : "";
      if (selectedPersona && selectedPersona.system_prompt) {
        systemPromptEl.value = selectedPersona.system_prompt;
      }
      if (selectedPersona) {
        jsonModeEl.checked = !!selectedPersona.json_mode;
      }
    });

    sendButtonEl.addEventListener("click", sendMessage);
    saveSettingsEl.addEventListener("click", saveSettings);
    clearHistoryEl.addEventListener("click", clearHistory);
    messageBoxEl.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        sendMessage();
      }
    });

    loadState().then(() => {
      setStatus("Ready.");
      messageBoxEl.focus();
    }).catch((error) => {
      setStatus(error.message);
    });
  </script>
</body>
</html>
"""


class ChatWebApp:
    def __init__(
        self,
        *,
        config_path: str,
        checkpoint_override: str = "",
        personas_path_override: str = "",
        persona_override: str = "",
        session_file_override: str = "",
    ) -> None:
        self.cfg = load_chat_config(config_path)
        self.personas_path = personas_path_override or self.cfg.personas_path
        self.personas = load_personas(self.personas_path)
        self.session_file = session_file_override or self.cfg.session_file
        self.session = load_session(self.session_file) if self.session_file else {"history": []}
        self.history = list(self.session.get("history", []))

        self.active_persona = resolve_persona(
            self.personas,
            persona_override or str(self.session.get("persona", "")).strip() or self.cfg.default_persona,
            fallback_system_prompt=self.cfg.system_prompt,
            fallback_json_mode=self.cfg.json_mode,
        )
        self.system_prompt = (
            str(self.session.get("system_prompt", "")).strip()
            or self.active_persona.system_prompt
        )
        self.json_mode = bool(self.session.get("json_mode", self.active_persona.json_mode or self.cfg.json_mode))
        self.lock = threading.Lock()

        self.model, self.tokenizer, self.model_cfg, self.device = load_chat_runtime(
            self.cfg,
            checkpoint_override=checkpoint_override,
        )

    def persist_session(self) -> None:
        if not self.session_file:
            return
        save_session(
            self.session_file,
            history=self.history,
            persona_name=self.active_persona.name,
            system_prompt=self.system_prompt,
            json_mode=self.json_mode,
        )

    def apply_settings(
        self,
        *,
        persona_name: str | None = None,
        system_prompt: str | None = None,
        json_mode: bool | None = None,
    ) -> None:
        if persona_name:
            self.active_persona = resolve_persona(
                self.personas,
                persona_name,
                fallback_system_prompt=self.system_prompt,
                fallback_json_mode=self.json_mode,
            )
            if system_prompt is None:
                system_prompt = self.active_persona.system_prompt
            if json_mode is None:
                json_mode = self.active_persona.json_mode
        if system_prompt is not None:
            self.system_prompt = system_prompt.strip() or self.active_persona.system_prompt
        if json_mode is not None:
            self.json_mode = bool(json_mode)
        self.persist_session()

    def clear_history(self) -> None:
        self.history.clear()
        self.persist_session()

    def chat(self, message: str) -> str:
        if self.cfg.tool_mode:
            result = run_agent_turn(
                model=self.model,
                tokenizer=self.tokenizer,
                model_cfg=self.model_cfg,
                device=self.device,
                system_prompt=self.system_prompt,
                history=self.history,
                user_message=message,
                max_history_turns=self.cfg.max_history_turns,
                registry=build_default_tool_registry(),
                max_tool_rounds=self.cfg.max_tool_rounds,
                json_mode=self.json_mode,
                thinking_mode=self.cfg.thinking_mode,
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
                max_new_tokens=self.cfg.max_new_tokens,
                repetition_penalty=self.cfg.repetition_penalty,
            )
            response = result.response
        else:
            response = generate_chat_response(
                model=self.model,
                tokenizer=self.tokenizer,
                model_cfg=self.model_cfg,
                device=self.device,
                system_prompt=self.system_prompt,
                history=self.history,
                user_message=message,
                max_history_turns=self.cfg.max_history_turns,
                json_mode=self.json_mode,
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
                max_new_tokens=self.cfg.max_new_tokens,
                repetition_penalty=self.cfg.repetition_penalty,
                thinking_mode=self.cfg.thinking_mode,
            )
        if not response:
            response = "(empty response)"
        self.history.append((message, response))
        self.persist_session()
        return response

    def state_payload(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "persona": self.active_persona.name,
            "system_prompt": self.system_prompt,
            "json_mode": self.json_mode,
            "thinking_mode": self.cfg.thinking_mode,
            "tool_mode": self.cfg.tool_mode,
            "session_file": str(Path(self.session_file).resolve()) if self.session_file else "",
            "history": [{"user": user, "assistant": assistant} for user, assistant in self.history],
            "personas": [
                {
                    "name": persona.name,
                    "description": persona.description,
                    "system_prompt": persona.system_prompt,
                    "json_mode": persona.json_mode,
                }
                for persona in sorted(self.personas.values(), key=lambda item: item.name)
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a lightweight local web UI for Tiny LLM chat.")
    parser.add_argument("--config", type=str, default="configs/chat.yaml", help="Path to chat YAML config.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path override.")
    parser.add_argument("--personas-path", type=str, default="", help="Optional personas JSON path override.")
    parser.add_argument("--persona", type=str, default="", help="Optional initial persona override.")
    parser.add_argument("--session-file", type=str, default="", help="Optional session JSON path override.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    return parser.parse_args()


def make_handler(app: ChatWebApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # pragma: no cover
            return

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("Request body must be valid JSON.") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object.")
            return payload

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send_html(HTML_PAGE)
                return
            if self.path == "/api/state":
                with app.lock:
                    self._send_json(app.state_payload())
                return
            if self.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT.value)
                self.end_headers()
                return
            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json()
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            try:
                if self.path == "/api/settings":
                    with app.lock:
                        app.apply_settings(
                            persona_name=str(payload.get("persona", "")).strip() or None,
                            system_prompt=str(payload.get("system_prompt", "")).strip() or None,
                            json_mode=payload.get("json_mode"),
                        )
                        self._send_json({"state": app.state_payload()})
                    return

                if self.path == "/api/clear":
                    with app.lock:
                        app.clear_history()
                        self._send_json({"state": app.state_payload()})
                    return

                if self.path == "/api/chat":
                    message = str(payload.get("message", "")).strip()
                    if not message:
                        self._send_json({"error": "message is required."}, status=HTTPStatus.BAD_REQUEST)
                        return
                    with app.lock:
                        app.apply_settings(
                            persona_name=str(payload.get("persona", "")).strip() or None,
                            system_prompt=str(payload.get("system_prompt", "")).strip() or None,
                            json_mode=payload.get("json_mode"),
                        )
                        response = app.chat(message)
                        self._send_json({"response": response, "state": app.state_payload()})
                    return
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    return Handler


def main() -> None:
    args = parse_args()
    app = ChatWebApp(
        config_path=args.config,
        checkpoint_override=args.checkpoint,
        personas_path_override=args.personas_path,
        persona_override=args.persona,
        session_file_override=args.session_file,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    url = f"http://{args.host}:{args.port}"
    print(f"[web] Tiny LLM chat available at {url}")
    print("[web] Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
