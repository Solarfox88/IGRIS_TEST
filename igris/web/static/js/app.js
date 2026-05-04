/* IGRIS_GPT — Agentic Engineering Console */
(function () {
  "use strict";

  // Helpers
  function $(sel) { return document.querySelector(sel); }
  function $$(sel) { return document.querySelectorAll(sel); }

  async function api(method, url, body) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body) opts.body = JSON.stringify(body);
    try {
      const r = await fetch(url, opts);
      return { ok: r.ok, status: r.status, data: await r.json() };
    } catch (e) {
      return { ok: false, status: 0, data: { error: e.message } };
    }
  }

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = String(s);
    return d.innerHTML;
  }

  function kvTable(obj) {
    let h = "<table>";
    for (const [k, v] of Object.entries(obj)) {
      const val = typeof v === "object" ? JSON.stringify(v) : String(v);
      h += "<tr><th>" + esc(k) + "</th><td>" + esc(val) + "</td></tr>";
    }
    return h + "</table>";
  }

  // Tab switching
  document.addEventListener("DOMContentLoaded", function () {
    $$(".tab").forEach(function (btn) {
      btn.addEventListener("click", function () {
        $$(".tab").forEach(function (b) { b.classList.remove("active"); });
        $$(".tab-pane").forEach(function (p) { p.classList.remove("active"); });
        btn.classList.add("active");
        var pane = $("#tab-" + btn.dataset.tab);
        if (pane) pane.classList.add("active");
      });
    });

    // Load initial data
    loadStatus();
    loadMission();
  });

  // Status header
  async function loadStatus() {
    var r = await api("GET", "/api/status");
    if (r.ok) {
      $("#header-status").textContent = "Online";
      $("#header-provider").textContent = r.data.provider + " / " + r.data.model;
    } else {
      $("#header-status").textContent = "Offline";
      $("#header-status").classList.add("error");
    }
  }

  // Mission Control
  async function loadMission() {
    var h = await api("GET", "/api/health");
    if (h.ok) {
      $("#mission-health").innerHTML = "<strong>Health</strong>" + kvTable(h.data);
    } else {
      $("#mission-health").innerHTML = '<span class="error">Health check failed</span>';
    }
    var rd = await api("GET", "/api/readiness");
    if (rd.ok) {
      $("#mission-readiness").innerHTML = "<strong>Readiness</strong>" + kvTable(rd.data);
    }
    var ctx = await api("GET", "/api/project/context");
    if (ctx.ok) {
      $("#mission-context").innerHTML = "<strong>Project Context</strong>" + kvTable(ctx.data);
    }
  }

  // Terminal
  (function () {
    var loaded = false;
    $$('.tab[data-tab="terminal"]').forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (!loaded) { loaded = true; loadTerminalCommands(); }
      });
    });
  })();

  async function loadTerminalCommands() {
    var r = await api("GET", "/api/terminal/commands");
    if (!r.ok) return;
    var container = $("#terminal-commands");
    container.innerHTML = "";
    r.data.commands.forEach(function (cmd) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "cmd-btn";
      b.textContent = cmd;
      b.setAttribute("aria-label", "Run " + cmd);
      b.addEventListener("click", function () { runTerminalCommand(cmd); });
      container.appendChild(b);
    });
  }

  async function runTerminalCommand(cmdId) {
    var out = $("#terminal-output");
    out.textContent = "Running " + cmdId + "...";
    var r = await api("POST", "/api/terminal/run", { command_id: cmdId });
    if (r.ok) {
      out.textContent = (r.data.stdout || "") + (r.data.stderr ? "\nSTDERR:\n" + r.data.stderr : "");
    } else {
      out.textContent = "Error: " + (r.data.detail || r.data.error || "unknown");
    }
  }

  // Files
  (function () {
    var loaded = false;
    $$('.tab[data-tab="files"]').forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (!loaded) { loaded = true; loadFileTree(); }
      });
    });
  })();

  async function loadFileTree() {
    var r = await api("GET", "/api/files/tree");
    if (!r.ok) return;
    var container = $("#file-tree");
    container.innerHTML = "";
    (r.data.tree || []).forEach(function (dir) {
      var dirEl = document.createElement("div");
      dirEl.className = "ft-dir";
      dirEl.textContent = dir.path === "." ? "/" : dir.path;
      container.appendChild(dirEl);
      (dir.entries || []).forEach(function (e) {
        var el = document.createElement("div");
        el.className = e.type === "dir" ? "ft-dir" : "ft-file";
        el.textContent = (e.type === "dir" ? "\uD83D\uDCC1 " : "\uD83D\uDCC4 ") + e.name;
        if (e.type === "file") {
          var filePath = (dir.path === "." ? "" : dir.path + "/") + e.name;
          el.addEventListener("click", function () { previewFile(filePath); });
        }
        container.appendChild(el);
      });
    });
  }

  async function previewFile(path) {
    var out = $("#file-preview");
    out.textContent = "Loading " + path + "...";
    var r = await api("GET", "/api/files/preview?path=" + encodeURIComponent(path));
    if (r.ok) {
      out.textContent = r.data.preview || "(empty)";
    } else {
      out.textContent = "Error: " + (r.data.detail || "unknown");
    }
  }

  // Git
  (function () {
    var loaded = false;
    $$('.tab[data-tab="git"]').forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (!loaded) { loaded = true; loadGit(); }
      });
    });
  })();

  async function loadGit() {
    var r = await api("GET", "/api/git/status");
    if (r.ok) {
      $("#git-info").innerHTML = kvTable(r.data);
    } else {
      $("#git-info").innerHTML = '<span class="error">Failed to load git status</span>';
    }
  }

  // Tests
  (function () {
    var btn = $("#btn-run-tests");
    if (btn) {
      btn.addEventListener("click", async function () {
        btn.disabled = true;
        var out = $("#test-output");
        out.textContent = "Running tests...";
        var r = await api("POST", "/api/tests/run");
        btn.disabled = false;
        if (r.ok) {
          out.textContent = (r.data.success ? "PASSED\n" : "FAILED\n") + (r.data.stdout || "") + (r.data.stderr ? "\nSTDERR:\n" + r.data.stderr : "");
        } else {
          out.textContent = "Error: " + (r.data.detail || "unknown");
        }
      });
    }
  })();

  // Logs
  (function () {
    var btn = $("#btn-refresh-logs");
    if (btn) {
      btn.addEventListener("click", async function () {
        var out = $("#logs-output");
        out.textContent = "Loading logs...";
        var r = await api("GET", "/api/logs");
        out.textContent = r.ok ? (r.data.logs || "(empty)") : "Error loading logs";
      });
    }
  })();

  // Agent Timeline
  (function () {
    var btn = $("#btn-refresh-timeline");
    if (btn) {
      btn.addEventListener("click", loadTimeline);
    }
    $$('.tab[data-tab="agent"]').forEach(function (b) {
      b.addEventListener("click", loadTimeline);
    });
  })();

  async function loadTimeline() {
    var container = $("#timeline-list");
    container.innerHTML = '<span class="loading">Loading...</span>';
    var r = await api("GET", "/api/agent/timeline");
    if (!r.ok) { container.innerHTML = '<span class="error">Failed</span>'; return; }
    var events = r.data.timeline || [];
    if (!events.length) { container.innerHTML = "<em>No events yet.</em>"; return; }
    container.innerHTML = "";
    events.forEach(function (ev) {
      var d = document.createElement("div");
      d.className = "timeline-event";
      d.innerHTML = '<span class="te-time">' + esc(ev.timestamp || "") + "</span> " + esc(ev.event || JSON.stringify(ev));
      container.appendChild(d);
    });
  }

  // Tasks
  (function () {
    var loaded = false;
    $$('.tab[data-tab="tasks"]').forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (!loaded) { loaded = true; }
        loadTasks();
      });
    });
    var form = $("#task-form");
    if (form) {
      form.addEventListener("submit", async function (e) {
        e.preventDefault();
        var inp = $("#task-input");
        var desc = inp.value.trim();
        if (!desc) return;
        await api("POST", "/api/tasks", { description: desc });
        inp.value = "";
        loadTasks();
      });
    }
  })();

  async function loadTasks() {
    var container = $("#task-list");
    container.innerHTML = '<span class="loading">Loading...</span>';
    var r = await api("GET", "/api/tasks");
    if (!r.ok) { container.innerHTML = '<span class="error">Failed</span>'; return; }
    var tasks = r.data.tasks || [];
    if (!tasks.length) { container.innerHTML = "<em>No tasks.</em>"; return; }
    container.innerHTML = "";
    tasks.forEach(function (t) {
      var d = document.createElement("div");
      d.className = "task-item";
      var statusCls = t.status || "pending";
      d.innerHTML =
        '<div><span class="task-status ' + esc(statusCls) + '">' + esc(t.status) + '</span> ' +
        esc(t.title || t.description) + '</div>' +
        '<div class="task-actions">' +
        (t.status === "pending" ? '<button type="button" data-action="complete" data-id="' + t.id + '">Complete</button><button type="button" data-action="block" data-id="' + t.id + '">Block</button>' : "") +
        '</div>';
      container.appendChild(d);
    });
    container.querySelectorAll("button[data-action]").forEach(function (btn) {
      btn.addEventListener("click", async function () {
        var action = btn.dataset.action;
        var id = btn.dataset.id;
        await api("POST", "/api/tasks/" + id + "/" + action, {});
        loadTasks();
      });
    });
  }

  // Safety
  (function () {
    $$('.tab[data-tab="safety"]').forEach(function (btn) {
      btn.addEventListener("click", loadSafety);
    });
  })();

  async function loadSafety() {
    var r = await api("GET", "/api/safety/status");
    if (r.ok) {
      $("#safety-info").innerHTML = kvTable(r.data);
    }
  }

  // Cost
  (function () {
    $$('.tab[data-tab="cost"]').forEach(function (btn) {
      btn.addEventListener("click", loadCost);
    });
  })();

  async function loadCost() {
    var s = await api("GET", "/api/cost/summary");
    if (s.ok) {
      $("#cost-summary").innerHTML = "<strong>Cost Summary</strong>" + kvTable(s.data);
    }
    var h = await api("GET", "/api/routing/history");
    if (h.ok) {
      var hist = h.data.history || [];
      $("#routing-history").innerHTML = "<strong>Routing History</strong><br>" + (hist.length ? hist.map(function (e) { return esc(e.provider + " / " + e.model); }).join("<br>") : "<em>No history.</em>");
    }
    var e = await api("GET", "/api/routing/explain");
    if (e.ok) {
      $("#routing-explain").innerHTML = "<strong>Routing Explanation</strong><br>" + esc(e.data.explanation || "");
    }
  }

  // A2A
  (function () {
    $$('.tab[data-tab="a2a"]').forEach(function (btn) {
      btn.addEventListener("click", loadA2A);
    });
  })();

  async function loadA2A() {
    var card = await api("GET", "/.well-known/agent-card.json");
    if (card.ok) {
      $("#a2a-card").innerHTML = "<strong>Agent Card</strong>" + kvTable(card.data);
    }
    var caps = await api("GET", "/api/a2a/capabilities");
    if (caps.ok) {
      var list = caps.data.capabilities || [];
      var h = "<strong>Capabilities</strong><table><tr><th>ID</th><th>Name</th><th>Risk</th></tr>";
      list.forEach(function (c) {
        h += "<tr><td>" + esc(c.id) + "</td><td>" + esc(c.name) + "</td><td>" + esc(c.risk) + "</td></tr>";
      });
      h += "</table>";
      $("#a2a-capabilities").innerHTML = h;
    }
  }

  // Chat
  (function () {
    var sessionId = null;
    var form = $("#chat-form");
    if (!form) return;
    form.addEventListener("submit", async function (e) {
      e.preventDefault();
      var inp = $("#chat-input");
      var msg = inp.value.trim();
      if (!msg) return;
      addMsg("user", msg);
      inp.value = "";
      if (!sessionId) {
        var s = await api("POST", "/api/sessions");
        if (s.ok) sessionId = s.data.id;
      }
      if (!sessionId) { addMsg("assistant", "Failed to create session"); return; }
      var r = await api("POST", "/api/sessions/" + sessionId + "/messages", { message: msg });
      if (r.ok) {
        addMsg("assistant", r.data.response);
      } else {
        addMsg("assistant", "Error: " + (r.data.detail || "unknown"));
      }
    });

    function addMsg(role, text) {
      var container = $("#chat-messages");
      var d = document.createElement("div");
      d.className = "msg msg-" + role;
      d.textContent = text;
      container.appendChild(d);
      container.scrollTop = container.scrollHeight;
    }
  })();
})();
