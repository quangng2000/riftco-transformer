"""Self-contained browser client for the local generation endpoint."""

from __future__ import annotations


CHAT_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>Riftco Transformer Chat</title>
  <style>
    :root {
      --ink: #f4f0e8;
      --muted: #9d9d94;
      --panel: rgba(25, 27, 27, 0.92);
      --panel-strong: #1f2221;
      --line: rgba(244, 240, 232, 0.12);
      --accent: #c8ff63;
      --accent-ink: #172008;
      --danger: #ff8d78;
      --shadow: 0 28px 90px rgba(0, 0, 0, 0.38);
    }

    * {
      box-sizing: border-box;
    }

    html,
    body {
      min-height: 100%;
    }

    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 18% 12%, rgba(200, 255, 99, 0.12), transparent 31rem),
        radial-gradient(circle at 85% 82%, rgba(99, 190, 255, 0.10), transparent 28rem),
        #0d0f0e;
      font-family:
        Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI",
        sans-serif;
    }

    body::before {
      position: fixed;
      inset: 0;
      z-index: -1;
      content: "";
      opacity: 0.16;
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px);
      background-size: 38px 38px;
      mask-image: linear-gradient(to bottom, black, transparent 86%);
    }

    button,
    textarea,
    input,
    select {
      font: inherit;
    }

    button {
      color: inherit;
    }

    .visually-hidden {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }

    .shell {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 18px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .mark {
      display: grid;
      width: 40px;
      height: 40px;
      place-items: center;
      border: 1px solid rgba(200, 255, 99, 0.35);
      border-radius: 12px;
      color: var(--accent);
      background: rgba(200, 255, 99, 0.08);
      font: 700 18px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    .brand h1,
    .brand p {
      margin: 0;
    }

    .brand h1 {
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }

    .brand p {
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
    }

    .runtime {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }

    .pill {
      display: inline-flex;
      min-height: 30px;
      align-items: center;
      gap: 7px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.025);
      font: 600 11px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }

    .dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #f7c968;
      box-shadow: 0 0 0 3px rgba(247, 201, 104, 0.12);
    }

    .dot.online {
      background: var(--accent);
      box-shadow: 0 0 0 3px rgba(200, 255, 99, 0.12);
    }

    .app {
      display: grid;
      min-height: min(760px, calc(100vh - 116px));
      grid-template-columns: minmax(0, 1fr) 270px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }

    .main {
      display: grid;
      min-width: 0;
      grid-template-rows: auto 1fr auto;
    }

    .intro {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 24px;
      padding: 26px 28px 20px;
      border-bottom: 1px solid var(--line);
    }

    .eyebrow {
      margin: 0 0 8px;
      color: var(--accent);
      font: 700 11px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .intro h2 {
      max-width: 620px;
      margin: 0;
      font-size: clamp(24px, 4vw, 39px);
      font-weight: 630;
      letter-spacing: -0.04em;
      line-height: 1.06;
    }

    .reset {
      flex: none;
      padding: 8px 10px;
      border: 0;
      border-radius: 9px;
      color: var(--muted);
      background: transparent;
      cursor: pointer;
    }

    .reset:hover,
    .reset:focus-visible {
      color: var(--ink);
      background: rgba(255, 255, 255, 0.06);
      outline: none;
    }

    .reset:disabled {
      opacity: 0.4;
      cursor: wait;
    }

    .messages {
      display: flex;
      min-height: 280px;
      flex-direction: column;
      gap: 20px;
      overflow-y: auto;
      padding: 30px 28px;
      scroll-behavior: smooth;
    }

    .message {
      display: grid;
      max-width: min(680px, 91%);
      grid-template-columns: 30px minmax(0, 1fr);
      gap: 11px;
      animation: enter 180ms ease-out both;
    }

    .message.user {
      align-self: flex-end;
      grid-template-columns: minmax(0, 1fr) 30px;
    }

    .message.user .avatar {
      grid-column: 2;
    }

    .message.user .bubble {
      grid-column: 1;
      grid-row: 1;
      border-color: rgba(200, 255, 99, 0.18);
      background: rgba(200, 255, 99, 0.08);
    }

    .avatar {
      display: grid;
      width: 30px;
      height: 30px;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 9px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.035);
      font: 700 11px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    .assistant .avatar {
      color: var(--accent);
      border-color: rgba(200, 255, 99, 0.22);
    }

    .bubble {
      padding: 13px 15px;
      border: 1px solid var(--line);
      border-radius: 4px 16px 16px;
      background: rgba(255, 255, 255, 0.035);
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.55;
    }

    .user .bubble {
      border-radius: 16px 4px 16px 16px;
    }

    .composer-wrap {
      padding: 18px 22px 22px;
      border-top: 1px solid var(--line);
      background: rgba(10, 12, 11, 0.42);
    }

    .composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: end;
      gap: 12px;
      padding: 10px 10px 10px 15px;
      border: 1px solid rgba(244, 240, 232, 0.18);
      border-radius: 17px;
      background: #131514;
      transition: border-color 150ms ease, box-shadow 150ms ease;
    }

    .composer:focus-within {
      border-color: rgba(200, 255, 99, 0.56);
      box-shadow: 0 0 0 4px rgba(200, 255, 99, 0.07);
    }

    textarea {
      width: 100%;
      min-height: 46px;
      max-height: 160px;
      resize: none;
      border: 0;
      outline: 0;
      color: var(--ink);
      background: transparent;
      line-height: 1.45;
    }

    textarea::placeholder {
      color: #73766f;
    }

    .send {
      display: inline-flex;
      height: 42px;
      align-items: center;
      gap: 8px;
      padding: 0 17px;
      border: 0;
      border-radius: 12px;
      color: var(--accent-ink);
      background: var(--accent);
      font-weight: 750;
      cursor: pointer;
    }

    .send:hover {
      filter: brightness(1.05);
    }

    .send:focus-visible {
      outline: 3px solid rgba(200, 255, 99, 0.28);
      outline-offset: 3px;
    }

    .send:disabled {
      opacity: 0.45;
      cursor: wait;
    }

    .hint {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin: 10px 4px 0;
      color: #777a73;
      font-size: 11px;
    }

    .sidebar {
      padding: 26px 22px;
      border-left: 1px solid var(--line);
      background: rgba(10, 12, 11, 0.34);
    }

    .sidebar h3 {
      margin: 0 0 18px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .settings {
      display: grid;
      gap: 17px;
      margin: 0;
    }

    .field {
      display: grid;
      gap: 7px;
    }

    .field-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    label,
    .field-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }

    output {
      color: var(--accent);
      font: 600 11px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }

    input[type="number"] {
      width: 100%;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 9px;
      outline: none;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.035);
    }

    input[type="number"]:focus {
      border-color: rgba(200, 255, 99, 0.48);
    }

    .rule {
      height: 1px;
      margin: 23px 0;
      background: var(--line);
    }

    .fact {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 9px 0;
      color: var(--muted);
      font-size: 11px;
    }

    .fact strong {
      max-width: 126px;
      overflow: hidden;
      color: var(--ink);
      font: 600 11px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace;
      text-align: right;
      text-overflow: ellipsis;
      text-transform: uppercase;
      white-space: nowrap;
    }

    .notice {
      margin: 23px 0 0;
      padding: 13px;
      border: 1px solid rgba(247, 201, 104, 0.2);
      border-radius: 11px;
      color: #c5bda9;
      background: rgba(247, 201, 104, 0.055);
      font-size: 11px;
      line-height: 1.5;
    }

    .status-error {
      color: var(--danger);
    }

    @keyframes enter {
      from {
        opacity: 0;
        transform: translateY(5px);
      }
    }

    @media (max-width: 760px) {
      .shell {
        width: min(100% - 18px, 680px);
        padding: 12px 0;
      }

      .topbar {
        align-items: flex-start;
      }

      .runtime .pill:not(:first-child) {
        display: none;
      }

      .app {
        min-height: calc(100vh - 78px);
        grid-template-columns: 1fr;
        border-radius: 18px;
      }

      .sidebar {
        border-top: 1px solid var(--line);
        border-left: 0;
      }

      .intro,
      .messages {
        padding-right: 18px;
        padding-left: 18px;
      }

      .intro h2 {
        font-size: 27px;
      }

      .composer-wrap {
        padding: 12px;
      }

      .hint span:last-child {
        display: none;
      }
    }

    @media (prefers-reduced-motion: reduce) {
      *,
      *::before,
      *::after {
        scroll-behavior: auto !important;
        animation-duration: 0.01ms !important;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <div class="mark" aria-hidden="true">RT</div>
        <div>
          <h1>Riftco Transformer</h1>
          <p>Local inference console</p>
        </div>
      </div>
      <div class="runtime" aria-label="Runtime status">
        <span class="pill"><span class="dot" id="status-dot"></span><span id="status-text">Connecting</span></span>
        <span class="pill" id="backend-pill">Backend · —</span>
        <span class="pill" id="cache-pill">Cache · —</span>
      </div>
    </header>

    <main class="app">
      <section class="main" aria-labelledby="chat-heading">
        <header class="intro">
          <div>
            <p class="eyebrow">Your runtime · your weights</p>
            <h2 id="chat-heading">Talk directly to the model you trained.</h2>
          </div>
          <button class="reset" id="reset" type="button">Clear</button>
        </header>

        <div class="messages" id="messages" role="log" aria-live="polite" aria-relevant="additions">
          <article class="message assistant" aria-label="Riftco Transformer message">
            <div class="avatar" aria-hidden="true">RT</div>
            <div class="bubble">Ready when you are. This is your small learning model, served locally through its configured KV-cache inference path.</div>
          </article>
        </div>

        <div class="composer-wrap">
          <form class="composer" id="composer">
            <label for="prompt" class="visually-hidden">Message</label>
            <textarea id="prompt" rows="1" maxlength="4000" placeholder="Ask the model something…" autocomplete="off" required></textarea>
            <button class="send" id="send" type="submit" disabled>
              <span id="send-label">Loading</span>
              <span aria-hidden="true">↗</span>
            </button>
          </form>
          <div class="hint">
            <span id="request-status" role="status">Enter to send · Shift+Enter for a new line</span>
            <span>Same-origin · no cloud API</span>
          </div>
        </div>
      </section>

      <aside class="sidebar" aria-label="Generation settings">
        <h3>Generation</h3>
        <div class="settings">
          <div class="field">
            <div class="field-row">
              <label for="tokens">New tokens</label>
              <output id="tokens-output" for="tokens">8</output>
            </div>
            <input id="tokens" type="range" min="1" max="64" value="8">
          </div>
          <div class="field">
            <div class="field-row">
              <label for="temperature">Temperature</label>
              <output id="temperature-output" for="temperature">0.0</output>
            </div>
            <input id="temperature" type="range" min="0" max="1.5" step="0.1" value="0">
          </div>
          <div class="field">
            <label for="seed">Sampling seed</label>
            <input id="seed" type="number" min="0" step="1" value="0" inputmode="numeric">
          </div>
        </div>

        <div class="rule"></div>
        <h3>Runtime</h3>
        <div class="fact"><span>Model stage</span><strong id="stage">—</strong></div>
        <div class="fact"><span>Context</span><strong id="context">—</strong></div>
        <div class="fact"><span>Vocabulary</span><strong id="vocabulary">—</strong></div>
        <div class="fact"><span>Artifact</span><strong id="artifact">—</strong></div>

        <p class="notice" id="context-notice">
          The transcript is visual only: each message is a fresh single-turn
          request using the lab’s PlainChat training format. Short prompts work
          best with this tiny framework smoke-test model.
        </p>
      </aside>
    </main>
  </div>

  <script>
    "use strict";

    const elements = {
      artifact: document.querySelector("#artifact"),
      backend: document.querySelector("#backend-pill"),
      cache: document.querySelector("#cache-pill"),
      composer: document.querySelector("#composer"),
      context: document.querySelector("#context"),
      contextNotice: document.querySelector("#context-notice"),
      dot: document.querySelector("#status-dot"),
      messages: document.querySelector("#messages"),
      prompt: document.querySelector("#prompt"),
      requestStatus: document.querySelector("#request-status"),
      reset: document.querySelector("#reset"),
      seed: document.querySelector("#seed"),
      send: document.querySelector("#send"),
      sendLabel: document.querySelector("#send-label"),
      stage: document.querySelector("#stage"),
      status: document.querySelector("#status-text"),
      temperature: document.querySelector("#temperature"),
      temperatureOutput: document.querySelector("#temperature-output"),
      tokens: document.querySelector("#tokens"),
      tokensOutput: document.querySelector("#tokens-output"),
      vocabulary: document.querySelector("#vocabulary"),
    };

    const initialMessage = elements.messages.firstElementChild.cloneNode(true);

    function setConnection(online, label) {
      elements.dot.classList.toggle("online", online);
      elements.status.textContent = label;
    }

    function addMessage(role, text) {
      const article = document.createElement("article");
      article.className = `message ${role}`;
      article.setAttribute(
        "aria-label",
        role === "user" ? "User message" : "Riftco Transformer message",
      );

      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.setAttribute("aria-hidden", "true");
      avatar.textContent = role === "user" ? "YOU" : "RT";

      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;

      article.append(avatar, bubble);
      elements.messages.append(article);
      elements.messages.scrollTop = elements.messages.scrollHeight;
    }

    function formatPrompt(message) {
      return `### User:\n${message}\n### Assistant:\n`;
    }

    async function readJson(response) {
      const value = await response.json();
      if (!response.ok) {
        throw new Error(value.detail || `Request failed (${response.status})`);
      }
      return value;
    }

    async function loadHealth() {
      try {
        const response = await fetch("/health", {
          headers: { "Accept": "application/json" },
        });
        const health = await readJson(response);
        setConnection(true, "Ready");
        elements.backend.textContent = `Backend · ${health.backend}`;
        elements.cache.textContent =
          `Cache · ${health.kv_cache} ${health.kv_cache_block_size}`;
        elements.stage.textContent = health.artifact_stage;
        elements.context.textContent = `${health.maximum_context} tokens`;
        elements.contextNotice.textContent =
          "The transcript is visual only: each message is a fresh PlainChat " +
          `request. Short prompts work best; only the latest ` +
          `${health.maximum_context} tokens are retained. This tiny artifact ` +
          "is a framework smoke test, not an assistant-quality model.";
        elements.vocabulary.textContent = health.vocabulary_size;
        elements.artifact.textContent = health.artifact_id.slice(0, 10);
        elements.artifact.title = health.artifact_id;
        elements.tokens.max = health.maximum_new_tokens;
        elements.tokens.value = Math.min(
          Number(elements.tokens.value),
          health.maximum_new_tokens,
        );
        elements.tokensOutput.value = elements.tokens.value;
        elements.send.disabled = false;
        elements.sendLabel.textContent = "Send";
      } catch (error) {
        setConnection(false, "Offline");
        elements.requestStatus.textContent = error.message;
        elements.requestStatus.classList.add("status-error");
      }
    }

    async function generate(message) {
      const temperature = Number(elements.temperature.value);
      const payload = {
        prompt: formatPrompt(message),
        max_new_tokens: Number(elements.tokens.value),
        temperature,
        seed: Number(elements.seed.value),
      };

      elements.send.disabled = true;
      elements.reset.disabled = true;
      elements.sendLabel.textContent = "Thinking";
      elements.requestStatus.textContent = "Running local generation…";
      elements.requestStatus.classList.remove("status-error");

      try {
        const response = await fetch("/v1/generate", {
          method: "POST",
          headers: {
            "Accept": "application/json",
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });
        const result = await readJson(response);
        const visible = result.generated_text.trim();
        addMessage("assistant", visible || "(The model produced no visible text.)");
        elements.requestStatus.textContent =
          `${result.generated_token_ids.length} tokens generated locally`;
      } catch (error) {
        addMessage("assistant", `Generation error: ${error.message}`);
        elements.requestStatus.textContent = error.message;
        elements.requestStatus.classList.add("status-error");
      } finally {
        elements.send.disabled = false;
        elements.reset.disabled = false;
        elements.sendLabel.textContent = "Send";
        elements.prompt.focus();
      }
    }

    elements.composer.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = elements.prompt.value.trim();
      if (!message || elements.send.disabled) {
        return;
      }
      addMessage("user", message);
      elements.prompt.value = "";
      elements.prompt.style.height = "auto";
      await generate(message);
    });

    elements.prompt.addEventListener("keydown", (event) => {
      if (
        event.key === "Enter" &&
        !event.shiftKey &&
        !event.isComposing
      ) {
        event.preventDefault();
        elements.composer.requestSubmit();
      }
    });

    elements.prompt.addEventListener("input", () => {
      elements.prompt.style.height = "auto";
      elements.prompt.style.height =
        `${Math.min(elements.prompt.scrollHeight, 160)}px`;
    });

    elements.tokens.addEventListener("input", () => {
      elements.tokensOutput.value = elements.tokens.value;
    });

    elements.temperature.addEventListener("input", () => {
      elements.temperatureOutput.value =
        Number(elements.temperature.value).toFixed(1);
    });

    elements.reset.addEventListener("click", () => {
      elements.messages.replaceChildren(initialMessage.cloneNode(true));
      elements.requestStatus.textContent =
        "Enter to send · Shift+Enter for a new line";
      elements.requestStatus.classList.remove("status-error");
      elements.prompt.focus();
    });

    loadHealth();
  </script>
</body>
</html>
""".encode("utf-8")


__all__ = ["CHAT_HTML"]
