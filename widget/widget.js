/*
 * Embeddable chat widget for the Technical-Document RAG backend.
 *
 * Usage on any website (single line):
 *   <script src="http://YOUR_HOST:8000/widget/widget.js"
 *           data-api="http://YOUR_HOST:8000/api"></script>
 *
 * It injects a floating button (bottom-right); clicking opens a chat panel that
 * POSTs to {data-api}/chat and renders the answer + source citations. Vanilla
 * JS, no framework, so it embeds anywhere. Non-streaming (one response per ask).
 */
(function () {
  "use strict";

  // Resolve the API base from this script tag's data-api (fallback: same origin).
  var thisScript = document.currentScript;
  var API = (thisScript && thisScript.getAttribute("data-api")) || "/api";
  API = API.replace(/\/+$/, ""); // trim trailing slash

  var CSS_HREF = (function () {
    // widget.css sits next to widget.js
    try {
      var src = thisScript.src;
      return src.replace(/widget\.js(\?.*)?$/, "widget.css");
    } catch (e) {
      return null;
    }
  })();

  // Conversation history kept in memory (sent with each request for follow-ups).
  var history = [];
  var busy = false;

  // ---- Inject stylesheet ----
  if (CSS_HREF) {
    var link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = CSS_HREF;
    document.head.appendChild(link);
  }

  // ---- Build DOM ----
  var root = document.createElement("div");
  root.id = "tdrag-widget";
  root.innerHTML = [
    '<button id="tdrag-launcher" aria-label="Open chat">💬</button>',
    '<div id="tdrag-panel" role="dialog" aria-label="Document assistant" hidden>',
    '  <div id="tdrag-header">',
    '    <span>📚 Document Assistant</span>',
    '    <button id="tdrag-close" aria-label="Minimize" title="Minimize">—</button>',
    "  </div>",
    '  <div id="tdrag-messages"></div>',
    '  <form id="tdrag-form">',
    '    <input id="tdrag-input" type="text" autocomplete="off"',
    '           placeholder="Ask about the documents…" />',
    '    <button id="tdrag-send" type="submit">Send</button>',
    "  </form>",
    "</div>",
  ].join("");
  document.body.appendChild(root);

  var launcher = root.querySelector("#tdrag-launcher");
  var panel = root.querySelector("#tdrag-panel");
  var closeBtn = root.querySelector("#tdrag-close");
  var messages = root.querySelector("#tdrag-messages");
  var form = root.querySelector("#tdrag-form");
  var input = root.querySelector("#tdrag-input");

  // ---- Helpers ----
  function esc(s) {
    var d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function addMessage(role, html) {
    var el = document.createElement("div");
    el.className = "tdrag-msg tdrag-" + role;
    el.innerHTML = html;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
    return el;
  }
  var greeted = false;

  function showGreeting() {
    // Fetch the configurable welcome message from the API (read fresh from
    // config.json server-side). Only shown once per widget session, and only
    // if non-empty (an empty message in config hides the greeting entirely).
    if (greeted) return;
    greeted = true;
    fetch(API + "/health")
      .then(function (r) { return r.ok ? r.json() : {}; })
      .then(function (info) {
        var msg = info && typeof info.welcome_message === "string"
          ? info.welcome_message.trim()
          : "";
        if (msg) addMessage("bot", esc(msg));
      })
      .catch(function () { /* no greeting if health is unreachable */ });
  }

  function openPanel() {
    panel.hidden = false;
    launcher.style.display = "none";
    input.focus();
    if (messages.childElementCount === 0) {
      showGreeting();
    }
  }
  function closePanel() {
    panel.hidden = true;
    launcher.style.display = "";
  }

  function renderAnswer(data) {
    var parts = [];
    if (data.low_confidence) {
      parts.push(
        '<div class="tdrag-lowconf">⚠️ Low confidence: limited matching ' +
          "content found. Verify against the sources.</div>"
      );
    }
    parts.push('<div class="tdrag-answer">' + esc(data.answer) + "</div>");

    if (data.sources && data.sources.length) {
      var src = data.sources
        .map(function (s) {
          var href = API + "/source/" + encodeURIComponent(s);
          return (
            '<a class="tdrag-source" href="' +
            href +
            '" target="_blank" rel="noopener" title="Open ' +
            esc(s) +
            '">' +
            esc(s) +
            "</a>"
          );
        })
        .join("");
      parts.push('<div class="tdrag-sources">Sources: ' + src + "</div>");
    }

    var botEl = addMessage("bot", parts.join(""));

    // Clickable follow-up suggestions.
    if (data.followups && data.followups.length) {
      var wrap = document.createElement("div");
      wrap.className = "tdrag-followups";
      data.followups.forEach(function (q) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "tdrag-followup";
        b.textContent = q;
        b.addEventListener("click", function () {
          input.value = q;
          form.dispatchEvent(new Event("submit", { cancelable: true }));
        });
        wrap.appendChild(b);
      });
      botEl.appendChild(wrap);
      messages.scrollTop = messages.scrollHeight;
    }
  }

  function ask(question) {
    if (busy || !question.trim()) return;
    busy = true;
    addMessage("user", esc(question));
    history.push({ role: "user", content: question });

    var thinking = addMessage("bot", '<span class="tdrag-dots">…</span>');

    fetch(API + "/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, history: history.slice(0, -1) }),
    })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        thinking.remove();
        renderAnswer(data);
        history.push({ role: "assistant", content: data.answer || "" });
      })
      .catch(function (err) {
        thinking.remove();
        addMessage(
          "bot",
          '<div class="tdrag-error">Sorry, the assistant is unavailable (' +
            esc(err.message) +
            ").</div>"
        );
      })
      .finally(function () {
        busy = false;
      });
  }

  // ---- Wire up events ----
  launcher.addEventListener("click", openPanel);
  closeBtn.addEventListener("click", closePanel);
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var q = input.value;
    input.value = "";
    ask(q);
  });
})();
