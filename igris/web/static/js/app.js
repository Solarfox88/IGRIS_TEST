/* IGRIS_GPT — Agentic Engineering Console */
(function () {
  "use strict";

  function $(sel) { return document.querySelector(sel); }
  function $$(sel) { return document.querySelectorAll(sel); }

  async function api(method, url, body) {
    var opts = { method: method, headers: { "Content-Type": "application/json" } };
    if (body) opts.body = JSON.stringify(body);
    try {
      var r = await fetch(url, opts);
      return { ok: r.ok, status: r.status, data: await r.json() };
    } catch (e) {
      return { ok: false, status: 0, data: { error: e.message } };
    }
  }

  function esc(s) {
    var d = document.createElement("div");
    d.textContent = String(s);
    return d.innerHTML;
  }

  function kvTable(obj) {
    var h = "<table>";
    for (var k in obj) {
      if (!obj.hasOwnProperty(k)) continue;
      var v = obj[k];
      var val = typeof v === "object" ? JSON.stringify(v) : String(v);
      h += "<tr><th>" + esc(k) + "</th><td>" + esc(val) + "</td></tr>";
    }
    return h + "</table>";
  }

  function statusBadge(status) {
    var cls = status === "completed" ? "completed" : status === "blocked" ? "blocked" : status === "running" ? "running" : "pending";
    return '<span class="task-status ' + cls + '">' + esc(status) + "</span>";
  }

  // Tab switching
  document.addEventListener("DOMContentLoaded", function () {
    // Primary tabs
    $$(".tab").forEach(function (btn) {
      btn.addEventListener("click", function () {
        $$(".tab").forEach(function (b) { b.classList.remove("active"); });
        $$(".tab-pane").forEach(function (p) { p.classList.remove("active"); });
        btn.classList.add("active");
        var pane = $("#tab-" + btn.dataset.tab);
        if (pane) pane.classList.add("active");
      });
    });

    // Sub-tab switching
    $$(".sub-tab").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var bar = btn.parentElement;
        bar.querySelectorAll(".sub-tab").forEach(function (b) { b.classList.remove("active"); });
        var container = bar.parentElement;
        container.querySelectorAll(".sub-tab-pane").forEach(function (p) { p.classList.remove("active"); });
        btn.classList.add("active");
        var pane = container.querySelector("#subtab-" + btn.dataset.subtab);
        if (pane) pane.classList.add("active");
      });
    });

    loadStatus();
    loadMission();
    loadDashboardExtras();
    var supRefresh = $("#btn-refresh-supervisor-monitor");
    if (supRefresh) {
      supRefresh.addEventListener("click", function () { loadSupervisorMonitor(); });
    }

    // Auto-refresh active tab every 15s (lightweight)
    setInterval(function () {
      var activeTab = $(".tab.active");
      if (!activeTab) return;
      var tab = activeTab.dataset.tab;
      if (tab === "dashboard") { loadMission(); loadDashboardExtras(); }
      else if (tab === "memory") loadTimeline();
      else if (tab === "tasks") { if (typeof loadLoopStatus === "function") loadLoopStatus(); }
      else if (tab === "safety") loadCost();
    }, 15000);
  });

  // Status header
  async function loadStatus() {
    var r = await api("GET", "/api/status");
    var rd = await api("GET", "/api/readiness");
    if (r.ok) {
      $("#header-status").textContent = "Online";
      $("#header-status").className = "";
      var providerText = r.data.provider + " / " + r.data.model;
      if (rd.ok && !rd.data.ollama_available) {
        providerText += " (fallback mode)";
      }
      $("#header-provider").textContent = providerText;
    } else {
      $("#header-status").textContent = "Offline";
      $("#header-status").className = "error";
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
    loadMissions();
  }

  async function loadDashboardExtras() {
    // Diagnostics summary
    var diag = await api("GET", "/api/diagnostics/summary");
    var diagEl = $("#dash-diagnostics-summary");
    if (diagEl) {
      if (diag.ok) {
        var d = diag.data;
        var html = '<div class="dash-summary">';
        html += '<span>Starvation: <strong>' + esc(d.starvation_detected ? "YES" : "OK") + '</strong></span>';
        html += '<span>Blocked: <strong>' + esc(String(d.blocked_task_count || 0)) + '</strong></span>';
        html += '<span>Health: <strong>' + esc(String(d.family_health_issues || 0)) + ' issues</strong></span>';
        html += '</div>';
        diagEl.innerHTML = html;
      } else {
        diagEl.innerHTML = '<span class="dim">Diagnostics unavailable</span>';
      }
    }

    // Loop summary
    var loop = await api("GET", "/api/loop/status");
    var loopEl = $("#dash-loop-info");
    if (loopEl) {
      if (loop.ok) {
        var ls = loop.data;
        var html = '<div class="dash-summary">';
        html += '<span>Steps: <strong>' + esc(String(ls.total_steps || 0)) + '</strong></span>';
        html += '<span>Last: <strong>' + esc(ls.last_action || "none") + '</strong></span>';
        html += '</div>';
        loopEl.innerHTML = html;
      } else {
        loopEl.innerHTML = '<span class="dim">Loop not started</span>';
      }
    }

    // Decision reports
    var reports = await api("GET", "/api/decision-reports");
    var reportsEl = $("#dash-reports");
    if (reportsEl) {
      if (reports.ok && reports.data.reports && reports.data.reports.length > 0) {
        var recent = reports.data.reports.slice(0, 3);
        var html = '';
        for (var i = 0; i < recent.length; i++) {
          var rp = recent[i];
          html += '<div class="dash-report-item">';
          html += '<span class="dim">' + esc(rp.id || "") + '</span> ';
          html += '<span>' + esc(rp.selected_task || rp.outcome || "report") + '</span>';
          html += '</div>';
        }
        reportsEl.innerHTML = html;
      } else {
        reportsEl.innerHTML = '<span class="dim">No decision reports yet</span>';
      }
    }

    await loadSupervisorMonitor();
  }

  async function loadSupervisorMonitor() {
    var monitorEl = $("#dash-supervisor-monitor");
    if (!monitorEl) return;
    monitorEl.innerHTML = "Loading supervisor runs...";

    var active = await api("GET", "/api/rank/runs/active");
    var audit = await api("GET", "/api/rank/audit/summary");
    if (!active.ok) {
      var errMsg = ((active.data || {}).detail || (active.data || {}).error || ("HTTP " + String(active.status || 0)));
      monitorEl.innerHTML = "Supervisor monitor unavailable: " + esc(String(errMsg));
      return;
    }

    var rows = [];
    rows.push("<div><strong>Rank / Mission Monitor</strong></div>");
    if (active.ok) {
      var runs = active.data.runs || [];
      if (!runs.length) {
        rows.push("<div><strong>Supervisor Runs:</strong> 0 active</div>");
        rows.push("<div>No active supervisor runs. Start a supervised mission or view recent audit history.</div>");
      } else {
        rows.push("<div><strong>Supervisor Runs:</strong> " + esc(String(runs.length)) + " active</div>");
        runs.slice(0, 3).forEach(function (run) {
          var stage = run.current_stage || "idle";
          var failedStage = run.failed_stage || "-";
          var next = run.next_action || "";
          var issueUrl = run.escalation_issue_url || "";
          var issueHtml = issueUrl ? ('<a href="' + esc(issueUrl) + '" target="_blank" rel="noopener noreferrer">issue</a>') : "-";
          rows.push(
            '<div class="dash-report-item">' +
            esc(run.run_id || "") +
            " | rank=" + esc(run.rank_id || "-") +
            " | status=" + esc(run.status || "") +
            " | outcome=" + esc(run.outcome || "-") +
            " | stage=" + esc(stage) +
            " | failed_stage=" + esc(failedStage) +
            " | failure=" + esc(run.failure_class || "-") +
            " | repairs=" + esc(String(run.repair_cycles_used || 0)) +
            " | api=" + esc(String(run.api_escalations_used || 0)) +
            " ($" + esc(String(run.api_budget_used_usd || 0)) + ")" +
            " | escalation_issue=" + issueHtml +
            " | audit_new=" + esc(String((((run.audit_summary || {}).counts || {})["audit-new"]) || 0)) +
            " | audit_reviewed=" + esc(String((((run.audit_summary || {}).counts || {})["audit-reviewed"]) || 0)) +
            " | audit_fixed=" + esc(String((((run.audit_summary || {}).counts || {})["audit-fixed"]) || 0)) +
            " | audit_deferred=" + esc(String((((run.audit_summary || {}).counts || {})["audit-deferred"]) || 0)) +
            " | next=" + esc(next) +
            "</div>"
          );
        });
      }
    }

    if (audit.ok) {
      var inMem = (((audit.data || {}).in_memory || {}).counts) || {};
      var persisted = (((audit.data || {}).persisted || {}).counts) || {};
      rows.push("<div><strong>Audit & Escalations</strong></div>");
      rows.push(
        "<div><strong>Audit (memory):</strong> " +
        "new=" + esc(String(inMem["audit-new"] || 0)) + ", " +
        "reviewed=" + esc(String(inMem["audit-reviewed"] || 0)) + ", " +
        "fixed=" + esc(String(inMem["audit-fixed"] || 0)) + ", " +
        "deferred=" + esc(String(inMem["audit-deferred"] || 0)) +
        "</div>"
      );
      rows.push(
        "<div><strong>Audit (persisted):</strong> " +
        "new=" + esc(String(persisted["audit-new"] || 0)) + ", " +
        "reviewed=" + esc(String(persisted["audit-reviewed"] || 0)) + ", " +
        "fixed=" + esc(String(persisted["audit-fixed"] || 0)) + ", " +
        "deferred=" + esc(String(persisted["audit-deferred"] || 0)) + ", " +
        "deferred_due=" + esc(String((((audit.data || {}).persisted || {}).deferred_due_count) || 0)) +
        "</div>"
      );
      var recent = ((audit.data || {}).recent_runs) || [];
      if (recent.length) {
        rows.push("<div><strong>Recent Runs:</strong></div>");
        recent.slice(0, 3).forEach(function (run) {
          rows.push(
            '<div class="dash-report-item">' +
            esc(run.run_id || "") +
            " | status=" + esc(run.status || "") +
            " | outcome=" + esc(run.outcome || "-") +
            " | failure=" + esc(run.failure_class || "-") +
            "</div>"
          );
        });
      } else {
        rows.push("<div><strong>Recent Runs:</strong> not available (in-memory history reset after restart).</div>");
      }
    }

    monitorEl.innerHTML = rows.join("") || '<span class="dim">No supervisor data</span>';
  }

  var _selectedMissionId = null;

  async function loadMissions() {
    var r = await api("GET", "/api/missions");
    var el = $("#mission-list");
    if (!el) return;
    if (!r.ok) { el.innerHTML = "<em>Failed to load missions</em>"; return; }
    var missions = r.data.missions || [];
    if (!missions.length) { el.innerHTML = "<em>No missions yet</em>"; return; }
    var h = "";
    missions.forEach(function (m) {
      h += '<div class="patch-item" data-mid="' + esc(m.id) + '" style="cursor:pointer">';
      h += "<strong>" + esc(m.title) + "</strong> " + statusBadge(m.status);
      h += "<br><small>" + (m.step_count || 0) + " steps | " + (m.task_ids || []).length + " tasks | " + esc(m.created_at) + "</small>";
      h += "</div>";
    });
    el.innerHTML = h;
    el.querySelectorAll(".patch-item").forEach(function (item) {
      item.addEventListener("click", function () {
        loadMissionDetail(item.dataset.mid);
      });
    });
  }

  async function loadMissionDetail(missionId) {
    _selectedMissionId = missionId;
    var r = await api("GET", "/api/missions/" + missionId);
    var el = $("#mission-detail");
    if (!el) return;
    if (!r.ok) { el.innerHTML = "<em>Mission not found</em>"; return; }
    var m = r.data;
    var h = "<strong>" + esc(m.title) + "</strong> " + statusBadge(m.status);
    h += "<br>" + esc(m.description || "");
    h += "<br><small>Steps: " + (m.step_count || 0) + " | Tasks: " + (m.task_ids || []).length + "</small>";
    h += '<div style="margin-top:.5rem">';
    if (m.status === "created") {
      h += '<button type="button" class="action-btn" id="btn-plan-mission">Generate Plan</button> ';
    }
    if (m.status === "planned") {
      h += '<button type="button" class="action-btn" id="btn-materialize-mission">Materialize Tasks</button> ';
    }
    h += '<button type="button" class="action-btn" id="btn-graph-mission">Show Graph</button>';
    h += "</div>";
    if (m.steps && m.steps.length) {
      h += "<h5>Plan Steps</h5>";
      m.steps.forEach(function (s, i) {
        h += '<div class="info-block" style="margin:.3rem 0;padding:.4rem .6rem">';
        h += "<strong>" + (i + 1) + ".</strong> " + esc(s.title) + " " + statusBadge(s.status);
        h += " <small>[" + esc(s.family) + "]</small>";
        if (s.dependencies && s.dependencies.length) {
          h += " <small>deps: " + s.dependencies.length + "</small>";
        }
        if (s.success_criteria && s.success_criteria.length) {
          h += "<br><small>Criteria: " + s.success_criteria.map(esc).join(", ") + "</small>";
        }
        h += "</div>";
      });
    }
    el.innerHTML = h;

    var planBtn = $("#btn-plan-mission");
    if (planBtn) {
      planBtn.addEventListener("click", async function () {
        var pr = await api("POST", "/api/missions/" + missionId + "/plan");
        if (pr.ok) { loadMissionDetail(missionId); loadMissions(); }
        else alert("Plan error: " + (pr.data.detail || "unknown"));
      });
    }
    var matBtn = $("#btn-materialize-mission");
    if (matBtn) {
      matBtn.addEventListener("click", async function () {
        var mr = await api("POST", "/api/missions/" + missionId + "/materialize-tasks");
        if (mr.ok) { loadMissionDetail(missionId); loadMissions(); }
        else alert("Materialize error: " + (mr.data.detail || "unknown"));
      });
    }
    var graphBtn = $("#btn-graph-mission");
    if (graphBtn) {
      graphBtn.addEventListener("click", function () { loadMissionGraph(missionId); });
    }
  }

  async function loadMissionGraph(missionId) {
    var r = await api("GET", "/api/missions/" + missionId + "/graph");
    var el = $("#mission-graph");
    if (!el) return;
    if (!r.ok) { el.innerHTML = "<em>Graph not available</em>"; return; }
    var g = r.data;
    var h = "<strong>" + esc(g.title) + "</strong> " + statusBadge(g.status);
    h += "<br><small>Nodes: " + g.nodes.length + " | Edges: " + g.edges.length + "</small>";
    h += '<div style="margin-top:.4rem">';
    g.nodes.forEach(function (n) {
      h += '<div style="display:inline-block;background:#21262d;border:1px solid #30363d;border-radius:4px;padding:.3rem .5rem;margin:.2rem;font-size:.78rem">';
      h += esc(n.title) + " " + statusBadge(n.status);
      h += "</div>";
    });
    if (g.edges.length) {
      h += "<br><small>Dependencies: ";
      g.edges.forEach(function (e) { h += esc(e.from.substring(0,6)) + "→" + esc(e.to.substring(0,6)) + " "; });
      h += "</small>";
    }
    h += "</div>";
    el.innerHTML = h;
  }

  // Mission form
  (function () {
    var form = $("#mission-form");
    if (!form) return;
    form.addEventListener("submit", async function (e) {
      e.preventDefault();
      var title = $("#mission-title").value.trim();
      var desc = $("#mission-desc").value.trim();
      if (!title) { alert("Mission title required"); return; }
      var r = await api("POST", "/api/missions", { title: title, description: desc });
      if (r.ok) {
        form.reset();
        loadMissions();
        loadMissionDetail(r.data.id);
      } else {
        alert("Error: " + (r.data.detail || r.data.error || "unknown"));
      }
    });
  })();

  // Mission refresh button
  (function () {
    var btn = $("#btn-refresh-missions");
    if (btn) btn.addEventListener("click", loadMissions);
  })();

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
      out.textContent = "Error " + r.status + ": " + (r.data.detail || "Access denied or file not found");
    }
  }

  // Git
  (function () {
    var loaded = false;
    $$('.tab[data-tab="git"]').forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (!loaded) { loaded = true; loadGit(); loadBranches(); }
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

  async function loadBranches() {
    var r = await api("GET", "/api/git/branches");
    if (r.ok) {
      var d = r.data;
      var html = "<strong>Current:</strong> " + esc(d.current || "unknown") + "<br>";
      html += "<strong>Branches:</strong> " + (d.branches || []).map(function(b) { return esc(b); }).join(", ");
      $("#git-branches").innerHTML = html;
    }
  }

  async function loadDiff(staged) {
    var url = "/api/git/diff" + (staged ? "?staged=true" : "");
    var r = await api("GET", url);
    if (r.ok) {
      var d = r.data;
      if (!d.diff) {
        $("#git-diff").innerHTML = "<em>No changes</em>";
        return;
      }
      var lines = d.diff.split("\n").map(function(l) {
        if (l.startsWith("+++") || l.startsWith("---")) return '<div class="diff-line diff-hdr">' + esc(l) + '</div>';
        if (l.startsWith("+")) return '<div class="diff-line diff-add">' + esc(l) + '</div>';
        if (l.startsWith("-")) return '<div class="diff-line diff-del">' + esc(l) + '</div>';
        if (l.startsWith("@@")) return '<div class="diff-line diff-hdr">' + esc(l) + '</div>';
        return '<div class="diff-line diff-ctx">' + esc(l) + '</div>';
      }).join("");
      var warn = d.secret_detected ? '<div class="error">⚠ Secret-like content detected and redacted</div>' : '';
      $("#git-diff").innerHTML = warn + lines;
    }
  }

  (function() {
    var el = $("#btn-refresh-git");
    if (el) el.addEventListener("click", function() { loadGit(); loadBranches(); });
    el = $("#btn-load-diff");
    if (el) el.addEventListener("click", function() { loadDiff(false); });
    el = $("#btn-load-staged-diff");
    if (el) el.addEventListener("click", function() { loadDiff(true); });

    el = $("#btn-git-safety");
    if (el) el.addEventListener("click", async function() {
      var r = await api("GET", "/api/git/safety-check");
      if (r.ok) {
        var d = r.data;
        var html = "<strong>Safe:</strong> " + (d.safe ? "✓ Yes" : "✗ No") + "<br>";
        if (d.staged_files && d.staged_files.length) html += "<strong>Staged:</strong> " + d.staged_files.map(esc).join(", ") + "<br>";
        if (d.warnings && d.warnings.length) html += '<div class="error">' + d.warnings.map(esc).join("<br>") + '</div>';
        if (d.secret_files && d.secret_files.length) html += "<strong>Secret files:</strong> " + d.secret_files.map(esc).join(", ") + "<br>";
        if (d.runtime_artifacts && d.runtime_artifacts.length) html += "<strong>Artifacts:</strong> " + d.runtime_artifacts.map(esc).join(", ");
        $("#git-safety").innerHTML = html;
      }
    });

    var branchForm = $("#git-branch-form");
    if (branchForm) branchForm.addEventListener("submit", async function(e) {
      e.preventDefault();
      var name = $("#git-branch-name").value.trim();
      if (!name) return;
      var r = await api("POST", "/api/git/branch", { name: name });
      if (r.ok && r.data.success) {
        $("#git-branch-name").value = "";
        loadGit();
        loadBranches();
      } else {
        alert("Error: " + (r.data.error || "Failed to create branch"));
      }
    });

    var commitForm = $("#git-commit-form");
    if (commitForm) commitForm.addEventListener("submit", async function(e) {
      e.preventDefault();
      var msg = $("#git-commit-msg").value.trim();
      if (!msg) return;
      var r = await api("POST", "/api/git/commit-proposal", { message: msg });
      if (r.ok) {
        var d = r.data;
        var html = "<strong>Message:</strong> " + esc(d.message) + "<br>";
        html += "<strong>Safe:</strong> " + (d.safe ? "✓ Yes" : "✗ No") + "<br>";
        if (d.files && d.files.length) html += "<strong>Files:</strong> " + d.files.map(esc).join(", ") + "<br>";
        if (d.warnings && d.warnings.length) html += '<div class="error">' + d.warnings.map(esc).join("<br>") + '</div>';
        if (d.blocked_files && d.blocked_files.length) html += "<strong>Blocked:</strong> " + d.blocked_files.map(esc).join(", ") + "<br>";
        if (d.secret_files && d.secret_files.length) html += "<strong>Secret files:</strong> " + d.secret_files.map(esc).join(", ");
        $("#git-commit-proposal").innerHTML = html;
      }
    });

    var prBtn = $("#btn-pr-summary");
    if (prBtn) prBtn.addEventListener("click", async function() {
      var r = await api("GET", "/api/git/pr-summary");
      if (r.ok) {
        var d = r.data;
        if (d.error) { $("#git-pr-summary").innerHTML = '<span class="error">' + esc(d.error) + '</span>'; return; }
        var html = "<strong>Branch:</strong> " + esc(d.branch) + " → " + esc(d.base) + "<br>";
        html += "<strong>Commits:</strong> " + (d.commit_count || 0) + "<br>";
        if (d.commits && d.commits.length) html += "<pre>" + d.commits.map(esc).join("\n") + "</pre>";
        if (d.summary) html += "<strong>Summary:</strong> " + esc(d.summary) + "<br>";
        if (d.stat) html += "<pre>" + esc(d.stat) + "</pre>";
        $("#git-pr-summary").innerHTML = html;
      }
    });
  })();

  // Tests
  (function () {
    var btn = $("#btn-run-tests");
    if (btn) {
      btn.addEventListener("click", async function () {
        btn.disabled = true;
        btn.textContent = "Running...";
        var out = $("#test-output");
        out.textContent = "Running tests...";
        var r = await api("POST", "/api/tests/run");
        btn.disabled = false;
        btn.textContent = "Run Tests";
        if (r.ok) {
          var prefix = r.data.success ? "PASSED\n" : "FAILED\n";
          out.textContent = prefix + (r.data.stdout || "") + (r.data.stderr ? "\nSTDERR:\n" + r.data.stderr : "");
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
    events.reverse().forEach(function (ev) {
      var d = document.createElement("div");
      d.className = "timeline-event";
      var severity = ev.severity || "info";
      var icon = severity === "warning" ? "\u26A0" : severity === "error" ? "\u274C" : "\u2022";
      var typeLabel = ev.type ? "[" + ev.type + "] " : "";
      var title = ev.title || ev.event || "";
      var detail = ev.detail || "";
      d.innerHTML = '<span class="te-icon">' + icon + '</span> ' +
        '<span class="te-type">' + esc(typeLabel) + '</span>' +
        '<strong>' + esc(title) + '</strong>' +
        (detail ? ' <span class="te-detail">' + esc(detail) + '</span>' : '') +
        (ev.timestamp ? ' <span class="te-time">' + esc(ev.timestamp) + '</span>' : '');
      container.appendChild(d);
    });
  }

  // Tasks
  (function () {
    var loaded = false;
    $$('.tab[data-tab="tasks"]').forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (!loaded) loaded = true;
        loadTasks();
      });
    });
    var form = $("#task-form");
    if (form) {
      form.addEventListener("submit", async function (e) {
        e.preventDefault();
        var desc = $("#task-input").value.trim();
        var title = $("#task-title-input") ? $("#task-title-input").value.trim() : "";
        if (!desc) return;
        var body = { description: desc };
        if (title) body.title = title;
        await api("POST", "/api/tasks", body);
        if ($("#task-input")) $("#task-input").value = "";
        if ($("#task-title-input")) $("#task-title-input").value = "";
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
      d.innerHTML =
        '<div class="task-header">' + statusBadge(t.status) +
        ' <strong>' + esc(t.title || t.description) + '</strong>' +
        ' <span class="task-meta">#' + t.id + ' | ' + esc(t.family || "other") + ' | ' + esc(t.source || "user") + '</span></div>' +
        (t.description && t.title ? '<div class="task-desc">' + esc(t.description) + '</div>' : '') +
        '<div class="task-actions">' +
        (t.status === "pending" || t.status === "running" ?
          '<button type="button" class="btn-sm" data-action="complete" data-id="' + t.id + '" aria-label="Complete task">Complete</button>' +
          '<button type="button" class="btn-sm btn-warn" data-action="block" data-id="' + t.id + '" aria-label="Block task">Block</button>' : "") +
        '</div>';
      container.appendChild(d);
    });
    container.querySelectorAll("button[data-action]").forEach(function (btn) {
      btn.addEventListener("click", async function () {
        await api("POST", "/api/tasks/" + btn.dataset.id + "/" + btn.dataset.action, {});
        loadTasks();
      });
    });
  }

  // Safety
  (function () {
    $$('.tab[data-tab="safety"]').forEach(function (btn) {
      btn.addEventListener("click", loadSafety);
    });
    var refreshBtn = $("#btn-refresh-safety");
    if (refreshBtn) refreshBtn.addEventListener("click", loadSafety);
  })();

  async function loadSafety() {
    var r = await api("GET", "/api/safety/status");
    if (r.ok) {
      var html = "<strong>Anti-Loop Status</strong>" + kvTable(r.data);
      // Add outcome router info
      var outcomes = await api("GET", "/api/outcome/recent");
      if (outcomes.ok && outcomes.data.outcomes && outcomes.data.outcomes.length) {
        html += "<strong>Recent Outcomes</strong><table><tr><th>Action</th><th>Reason</th></tr>";
        outcomes.data.outcomes.forEach(function (o) {
          html += "<tr><td>" + esc(o.next_action || "none") + "</td><td>" + esc(o.reason || "") + "</td></tr>";
        });
        html += "</table>";
      }
      $("#safety-info").innerHTML = html;
    }
    loadReports();
  }

  // Cost
  (function () {
    $$('.tab[data-tab="cost"]').forEach(function (btn) {
      btn.addEventListener("click", loadCost);
    });
    var refreshBtn = $("#btn-refresh-cost");
    if (refreshBtn) refreshBtn.addEventListener("click", loadCost);
    var estBtn = $("#btn-estimate-route");
    if (estBtn) estBtn.addEventListener("click", loadRouteEstimate);
  })();

  async function loadCost() {
    // Availability
    var av = await api("GET", "/api/routing/availability");
    if (av.ok) {
      var d = av.data;
      var html = "";
      var providers = ["ollama", "openai", "vastai"];
      providers.forEach(function (p) {
        var info = d[p] || {};
        var dot = info.available ? "ok" : "off";
        html += '<div class="provider-card"><span class="status-dot ' + dot + '"></span>';
        html += "<strong>" + esc(p) + "</strong>";
        if (info.model) html += " <small>(" + esc(info.model) + ")</small>";
        html += " <small>$" + (info.cost_per_call || 0) + "/call</small>";
        if (info.auto_provision === false) html += " <small>[no auto]</small>";
        html += "</div>";
      });
      $("#cost-availability").innerHTML = html;
    }
    // Budget
    var bg = await api("GET", "/api/cost/budget");
    if (bg.ok) {
      var b = bg.data;
      var pct = Math.min(b.usage_percent || 0, 100);
      var cls = b.exceeded ? "exceeded" : b.warning ? "warn" : "ok";
      var bhtml = '<div class="budget-bar"><div class="budget-fill ' + cls + '" style="width:' + pct + '%"></div></div>';
      bhtml += "<small>$" + (b.spent || 0) + " / $" + (b.max_session_cost || 0) + " (" + pct + "%)</small>";
      if (b.warning) bhtml += ' <span class="error"> Budget warning</span>';
      $("#cost-budget").innerHTML = bhtml;
    }
    // Summary
    var s = await api("GET", "/api/cost/summary");
    if (s.ok) {
      var sd = s.data;
      $("#cost-summary").innerHTML = kvTable({
        total_calls: sd.total_calls,
        local_calls: sd.local_calls,
        fallback_calls: sd.fallback_calls,
        estimated_cost_total: "$" + sd.estimated_cost_total,
        last_provider: sd.last_provider || "none",
      });
    }
    // Explain
    var e = await api("GET", "/api/routing/explain");
    if (e.ok) {
      $("#routing-explain").innerHTML = esc(e.data.explanation || "No routing decision yet.");
    }
  }

  async function loadRouteEstimate() {
    var r = await api("POST", "/api/routing/estimate", {task_type: "chat", complexity: "low"});
    if (r.ok) {
      var d = r.data;
      var html = kvTable({
        recommended_provider: d.recommended_provider,
        model: d.model,
        reason: d.reason,
        estimated_cost: "$" + d.estimated_cost,
        budget_remaining: "$" + d.budget_remaining,
        would_exceed_budget: d.would_exceed_budget,
      });
      $("#cost-estimate").innerHTML = html;
    } else {
      $("#cost-estimate").innerHTML = '<span class="error">Failed to estimate route</span>';
    }
  }

  // A2A
  (function () {
    $$('.tab[data-tab="a2a"]').forEach(function (btn) {
      btn.addEventListener("click", loadA2A);
    });
    var refreshBtn = $("#btn-refresh-a2a");
    if (refreshBtn) refreshBtn.addEventListener("click", loadA2A);
  })();

  async function loadA2A() {
    var card = await api("GET", "/.well-known/agent-card.json");
    if (card.ok) {
      $("#a2a-card").innerHTML = kvTable(card.data);
    }
    var caps = await api("GET", "/api/a2a/capabilities");
    if (caps.ok) {
      var list = caps.data.capabilities || [];
      var h = "<table><tr><th>ID</th><th>Name</th><th>Risk</th><th>Safe</th></tr>";
      list.forEach(function (c) {
        h += "<tr><td>" + esc(c.id) + "</td><td>" + esc(c.name) + "</td><td>" + esc(c.risk) + "</td><td>" + esc(c.safe) + "</td></tr>";
      });
      h += "</table>";
      $("#a2a-capabilities").innerHTML = h;
    }
    // A2A Store tasks
    var tasks = await api("GET", "/api/a2a/store/tasks");
    if (tasks.ok) {
      var tl = tasks.data.tasks || [];
      if (tl.length) {
        var th = "<table><tr><th>ID</th><th>Title</th><th>Status</th></tr>";
        tl.forEach(function (t) {
          th += "<tr><td>" + esc(t.id) + "</td><td>" + esc(t.title || t.description || "") + "</td><td>" + statusBadge(t.status) + "</td></tr>";
        });
        th += "</table>";
        $("#a2a-tasks").innerHTML = th;
      } else {
        $("#a2a-tasks").innerHTML = "<em>No A2A tasks.</em>";
      }
    } else {
      $("#a2a-tasks").innerHTML = "<em>No A2A tasks.</em>";
    }
  }

  // Chat
  (function () {
    var sessionId = null;
    var form = $("#chat-form");
    if (!form) return;
    var chatContainer = $("#chat-messages");
    var userNearBottom = true;

    // Track scroll position for auto-scroll
    if (chatContainer) {
      chatContainer.addEventListener("scroll", function () {
        var threshold = 60;
        userNearBottom = (chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight) < threshold;
      });
    }

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
      addMsg("assistant", "...", "typing");
      var r = await api("POST", "/api/sessions/" + sessionId + "/messages", { message: msg });
      removeTyping();
      if (r.ok) {
        var meta = {
          provider: r.data.provider,
          model: r.data.model,
          latency_ms: r.data.latency_ms,
          fallback_used: r.data.fallback_used,
          intent: r.data.intent_detected || null,
          actions: r.data.suggested_actions || [],
        };
        addMsg("assistant", r.data.response, null, meta);
      } else {
        addMsg("assistant", "Error: " + (r.data.detail || "unknown"));
      }
    });

    // Safe markdown renderer — no raw HTML injection
    function renderMarkdown(text) {
      if (!text) return "";
      // Escape HTML first to prevent XSS
      var escaped = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

      // Code blocks (```...```)
      escaped = escaped.replace(/```(\w*)\n?([\s\S]*?)```/g, function (m, lang, code) {
        return '<pre><code class="lang-' + lang + '">' + code.trim() + '</code><button class="copy-btn" onclick="igrisCopyCode(this)">copy</button></pre>';
      });

      // Inline code (`...`)
      escaped = escaped.replace(/`([^`\n]+)`/g, '<code>$1</code>');

      // Bold (**...**)
      escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

      // Split into blocks by double newline
      var blocks = escaped.split(/\n\n+/);
      var html = "";
      for (var i = 0; i < blocks.length; i++) {
        var block = blocks[i].trim();
        if (!block) continue;
        if (block.startsWith("<pre>")) {
          html += block;
        } else if (/^[-*]\s/.test(block) || /^\n?[-*]\s/.test(block)) {
          // Bullet list
          var items = block.split(/\n/).filter(function (l) { return l.trim(); });
          html += "<ul>";
          for (var j = 0; j < items.length; j++) {
            html += "<li>" + items[j].replace(/^[-*]\s+/, "") + "</li>";
          }
          html += "</ul>";
        } else if (/^\d+\.\s/.test(block)) {
          // Numbered list
          var items2 = block.split(/\n/).filter(function (l) { return l.trim(); });
          html += "<ol>";
          for (var k = 0; k < items2.length; k++) {
            html += "<li>" + items2[k].replace(/^\d+\.\s+/, "") + "</li>";
          }
          html += "</ol>";
        } else {
          // Handle single newlines as line items within a paragraph-like block
          var lines = block.split(/\n/);
          if (lines.length > 1 && lines.every(function (l) { return /^[-*]\s/.test(l.trim()); })) {
            html += "<ul>";
            for (var m = 0; m < lines.length; m++) {
              html += "<li>" + lines[m].replace(/^[-*]\s+/, "") + "</li>";
            }
            html += "</ul>";
          } else {
            html += "<p>" + block.replace(/\n/g, "<br>") + "</p>";
          }
        }
      }
      return html;
    }

    function addMsg(role, text, cls, meta) {
      var container = $("#chat-messages");
      var d = document.createElement("div");
      d.className = "msg msg-" + role + (cls ? " " + cls : "");

      if (role === "assistant" && !cls) {
        // Render markdown for assistant messages
        d.innerHTML = renderMarkdown(text);
        // Add suggested action buttons if available
        if (meta && meta.actions && meta.actions.length > 0) {
          var actionsDiv = document.createElement("div");
          actionsDiv.className = "suggested-actions";
          for (var ai = 0; ai < meta.actions.length; ai++) {
            var act = meta.actions[ai];
            var btn = document.createElement("button");
            btn.className = "action-card" + (act.approval_required ? " action-gated" : "");
            btn.innerHTML = '<span class="action-label">' + escapeHtml(act.label) + '</span>' +
              '<span class="action-desc">' + escapeHtml(act.description) + '</span>' +
              (act.approval_required ? '<span class="action-lock">requires approval</span>' : '');
            btn.dataset.endpoint = act.endpoint;
            btn.dataset.method = act.method || "GET";
            btn.dataset.payload = act.payload ? JSON.stringify(act.payload) : "";
            btn.addEventListener("click", handleActionClick);
            actionsDiv.appendChild(btn);
          }
          d.appendChild(actionsDiv);
        }
        // Add metadata line if available
        if (meta && meta.provider) {
          var metaDiv = document.createElement("div");
          metaDiv.className = "msg-meta";
          var provLabel = meta.provider === "igris_personality" ? "IGRIS" :
                          meta.provider === "deterministic" ? "fallback" :
                          meta.provider + "/" + meta.model;
          metaDiv.innerHTML = '<span class="meta-provider">' + escapeHtml(provLabel) + '</span>' +
            (meta.latency_ms != null ? '<span>' + meta.latency_ms + 'ms</span>' : '') +
            (meta.intent ? '<span>intent: ' + escapeHtml(meta.intent) + '</span>' : '');
          d.appendChild(metaDiv);
        }
      } else {
        d.textContent = text;
      }

      container.appendChild(d);
      if (userNearBottom) {
        container.scrollTop = container.scrollHeight;
      }
    }

    function escapeHtml(str) {
      return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    async function handleActionClick(e) {
      var btn = e.currentTarget;
      var endpoint = btn.dataset.endpoint;
      var method = btn.dataset.method || "GET";
      var payloadStr = btn.dataset.payload;

      btn.disabled = true;
      btn.classList.add("action-loading");

      var payload = payloadStr ? JSON.parse(payloadStr) : null;
      var r;
      if (method === "POST") {
        r = await api("POST", endpoint, payload || {});
      } else {
        r = await api("GET", endpoint);
      }

      btn.disabled = false;
      btn.classList.remove("action-loading");

      if (r.ok) {
        var resultText = JSON.stringify(r.data, null, 2);
        if (resultText.length > 2000) resultText = resultText.substring(0, 2000) + "\n...";
        addMsg("assistant", "```json\n" + resultText + "\n```");
      } else {
        addMsg("assistant", "Error: " + (r.data.detail || "request failed"));
      }
    }

    function removeTyping() {
      var el = $(".msg.typing");
      if (el) el.remove();
    }
  })();

  // Global copy function for code blocks
  window.igrisCopyCode = function (btn) {
    var pre = btn.parentElement;
    var code = pre.querySelector("code");
    if (code) {
      navigator.clipboard.writeText(code.textContent).then(function () {
        btn.textContent = "copied!";
        setTimeout(function () { btn.textContent = "copy"; }, 1500);
      });
    }
  };

  // Teacher Remediation button
  (function () {
    var btn = $("#btn-teacher-remediate");
    if (!btn) return;
    btn.addEventListener("click", async function () {
      btn.disabled = true;
      btn.textContent = "Analyzing...";
      var out = $("#teacher-output");
      out.textContent = "Building teacher payload...";
      var r = await api("POST", "/api/teacher/remediate", { create: false });
      btn.disabled = false;
      btn.textContent = "Ask Teacher";
      if (!r.ok) {
        out.textContent = "Error: " + (r.data.detail || "unknown");
        return;
      }
      var d = r.data;
      var html = "<strong>Proposed Task</strong>" + kvTable(d.proposed_task || {});
      if (d.validation) {
        html += "<strong>Validation</strong>" + kvTable(d.validation);
      }
      html += '<br><button type="button" id="btn-teacher-create" class="cmd-btn" aria-label="Create remediation task">Create This Task</button>';
      out.innerHTML = html;

      var createBtn = $("#btn-teacher-create");
      if (createBtn) {
        createBtn.addEventListener("click", async function () {
          createBtn.disabled = true;
          var cr = await api("POST", "/api/teacher/remediate", { create: true });
          if (cr.ok && cr.data.created_task_id) {
            out.innerHTML += '<br><span style="color:#4caf50">Task #' + cr.data.created_task_id + ' created!</span>';
          } else {
            out.innerHTML += '<br><span class="error">Could not create task (validation failed or no proposal)</span>';
          }
        });
      }
    });
  })();

  // Reports (loaded with Safety tab)
  (function () {
    // loadReports is called from loadSafety
  })();

  async function loadReports() {
    var container = $("#reports-list");
    if (!container) return;
    container.innerHTML = '<span class="loading">Loading...</span>';
    var r = await api("GET", "/api/reports/recent");
    if (!r.ok) { container.innerHTML = '<span class="error">Failed</span>'; return; }
    var reports = r.data.reports || [];
    if (!reports.length) { container.innerHTML = "<em>No reports yet.</em>"; return; }
    var html = "<table><tr><th>Command</th><th>Success</th><th>Duration</th><th>Time</th></tr>";
    reports.reverse().forEach(function (rp) {
      html += "<tr><td>" + esc(rp.command_id) + "</td><td>" +
        (rp.success ? "Yes" : "No") + "</td><td>" +
        esc(rp.duration_ms + "ms") + "</td><td>" +
        esc(rp.started_at || "") + "</td></tr>";
    });
    html += "</table>";
    container.innerHTML = html;
  }

  // ---- Memory ----
  async function loadMemory() {
    var cEl = $("#memory-constraints");
    var dEl = $("#memory-decisions");
    var fEl = $("#memory-failures");
    var cr = await api("GET", "/api/memory/saturation");
    if (cr.ok && cEl) {
      var c = cr.data.constraints || {};
      var h = "<strong>Recommendation:</strong> " + esc(c.recommendation || "No constraints");
      h += "<br><small>Saturated: " + (c.saturated_families || []).map(esc).join(", ");
      h += " | Failures: " + (c.recent_failure_count || 0);
      h += " | Decisions: " + (c.recent_decision_count || 0);
      h += " | Remediations: " + (c.remediation_count || 0) + "</small>";
      if ((c.avoid_families || []).length) {
        h += '<br><span class="task-status blocked">Avoid: ' + c.avoid_families.map(esc).join(", ") + "</span>";
      }
      cEl.innerHTML = h;
    }
    var dr = await api("GET", "/api/memory/decisions?limit=10");
    if (dr.ok && dEl) {
      var evts = dr.data.events || [];
      if (!evts.length) { dEl.innerHTML = "<em>No decisions yet</em>"; }
      else {
        var h2 = "";
        evts.forEach(function (e) {
          h2 += '<div class="info-block" style="margin:.2rem 0;padding:.3rem .5rem">';
          h2 += statusBadge(e.outcome || "pending") + " <strong>" + esc(e.title) + "</strong>";
          h2 += " <small>[" + esc(e.family || "—") + "]</small>";
          if (e.reason) h2 += "<br><small>" + esc(e.reason) + "</small>";
          h2 += "</div>";
        });
        dEl.innerHTML = h2;
      }
    }
    var fr = await api("GET", "/api/memory/failures?limit=10");
    if (fr.ok && fEl) {
      var fevts = fr.data.events || [];
      if (!fevts.length) { fEl.innerHTML = "<em>No failures recorded</em>"; }
      else {
        var h3 = "";
        fevts.forEach(function (e) {
          h3 += '<div class="info-block" style="margin:.2rem 0;padding:.3rem .5rem">';
          h3 += '<span class="task-status blocked">failure</span> <strong>' + esc(e.title) + "</strong>";
          h3 += " <small>[" + esc(e.family || "—") + "]</small>";
          if (e.reason) h3 += "<br><small>" + esc(e.reason) + "</small>";
          h3 += "</div>";
        });
        fEl.innerHTML = h3;
      }
    }
  }

  (function () {
    var loaded = false;
    $$('.tab[data-tab="memory"]').forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (!loaded) { loaded = true; loadMemory(); }
      });
    });
    var refreshBtn = $("#btn-refresh-memory");
    if (refreshBtn) refreshBtn.addEventListener("click", loadMemory);

    var form = $("#memory-event-form");
    if (form) {
      form.addEventListener("submit", async function (e) {
        e.preventDefault();
        var evType = $("#memory-event-type").value;
        var title = $("#memory-event-title").value.trim();
        var family = $("#memory-event-family").value.trim();
        var desc = $("#memory-event-desc").value.trim();
        var r = await api("POST", "/api/memory/events", {
          event_type: evType, title: title, family: family, description: desc
        });
        if (r.ok) { form.reset(); loadMemory(); }
        else alert("Error: " + (r.data.detail || "unknown"));
      });
    }
  })();

  // ---- Loop ----
  async function loadLoopStatus() {
    var sEl = $("#loop-status");
    var rEl = $("#loop-recent");
    var sr = await api("GET", "/api/loop/status");
    if (sr.ok && sEl) {
      var s = sr.data;
      var h = "<strong>Running:</strong> " + (s.running ? "Yes" : "No");
      h += " | <strong>Steps:</strong> " + (s.steps_completed || 0) + "/" + (s.max_steps || 0);
      if (s.stopped_reason) h += '<br><span class="task-status blocked">' + esc(s.stopped_reason) + "</span>";
      if (s.started_at) h += "<br><small>Started: " + esc(s.started_at) + "</small>";
      if (s.finished_at) h += " <small>Finished: " + esc(s.finished_at) + "</small>";
      sEl.innerHTML = h;
    }
    var rr = await api("GET", "/api/loop/recent?limit=10");
    if (rr.ok && rEl) {
      var steps = rr.data.steps || [];
      if (!steps.length) { rEl.innerHTML = "<em>No steps executed yet</em>"; }
      else {
        var h2 = "";
        steps.forEach(function (s) {
          h2 += '<div class="info-block" style="margin:.2rem 0;padding:.3rem .5rem">';
          h2 += "<strong>#" + s.step_number + "</strong> ";
          h2 += statusBadge(s.outcome || "pending") + " ";
          h2 += esc(s.action_type || "") + " ";
          if (s.task_title) h2 += "— " + esc(s.task_title);
          if (s.action_detail) h2 += "<br><small>" + esc(s.action_detail) + "</small>";
          if (s.reason) h2 += "<br><small>" + esc(s.reason) + "</small>";
          h2 += "</div>";
        });
        rEl.innerHTML = h2;
      }
    }
  }

  async function runLoopSteps(n) {
    var sEl = $("#loop-status");
    if (sEl) sEl.innerHTML = '<span class="loading">Running ' + n + ' step(s)...</span>';
    var r;
    if (n === 1) {
      r = await api("POST", "/api/loop/step");
    } else {
      r = await api("POST", "/api/loop/run", { max_steps: n });
    }
    loadLoopStatus();
  }

  (function () {
    var loaded = false;
    $$('.tab[data-tab="loop"]').forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (!loaded) { loaded = true; loadLoopStatus(); }
      });
    });
    var refreshBtn = $("#btn-refresh-loop");
    if (refreshBtn) refreshBtn.addEventListener("click", loadLoopStatus);
    var stepBtn = $("#btn-loop-step");
    if (stepBtn) stepBtn.addEventListener("click", function () { runLoopSteps(1); });
    var run3Btn = $("#btn-loop-run3");
    if (run3Btn) run3Btn.addEventListener("click", function () { runLoopSteps(3); });
    var run5Btn = $("#btn-loop-run5");
    if (run5Btn) run5Btn.addEventListener("click", function () { runLoopSteps(5); });
  })();

  // ---- Patches ----
  (function () {
    var loaded = false;
    $$('.tab[data-tab="patches"]').forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (!loaded) { loaded = true; loadPatches(); }
      });
    });
    var refreshBtn = $("#btn-refresh-patches");
    if (refreshBtn) refreshBtn.addEventListener("click", loadPatches);
  })();

  async function loadPatches() {
    var container = $("#patches-list");
    if (!container) return;
    container.innerHTML = '<span class="loading">Loading...</span>';
    var r = await api("GET", "/api/patches");
    if (!r.ok) { container.innerHTML = '<span class="error">Failed to load patches</span>'; return; }
    var patches = r.data.patches || [];
    if (!patches.length) { container.innerHTML = "<em>No patch proposals yet.</em>"; return; }
    var html = "";
    patches.forEach(function (p) {
      html += '<div class="patch-item" data-patch-id="' + esc(p.id) + '">' +
        '<span class="pi-title">' + esc(p.title) + '</span> ' +
        '<span class="patch-status ' + esc(p.status) + '">' + esc(p.status) + '</span>' +
        '<div class="pi-meta">' + esc(p.file_count) + ' file(s) | risk: ' + esc(p.risk) + ' | ' + esc(p.created_at) + '</div>' +
        '</div>';
    });
    container.innerHTML = html;
    $$(".patch-item").forEach(function (el) {
      el.addEventListener("click", function () { loadPatchDetail(el.dataset.patchId); });
    });
  }

  async function loadPatchDetail(id) {
    var detail = $("#patch-detail");
    var diffBox = $("#patch-diff");
    var actions = $("#patch-actions");
    if (!detail) return;
    detail.innerHTML = '<span class="loading">Loading...</span>';
    if (diffBox) diffBox.innerHTML = "";
    if (actions) actions.innerHTML = "";

    var r = await api("GET", "/api/patches/" + id);
    if (!r.ok) { detail.innerHTML = '<span class="error">Failed to load proposal</span>'; return; }
    var p = r.data;

    var html = "<strong>" + esc(p.title) + "</strong> " +
      '<span class="patch-status ' + esc(p.status) + '">' + esc(p.status) + '</span>' +
      "<p>" + esc(p.description) + "</p>" +
      "<p>Risk: " + esc(p.risk) + " | Files: " + p.files.length + "</p>";
    if (p.validation) {
      html += "<p><strong>Validation:</strong> " + (p.validation.valid ? "PASSED" : "FAILED") + " (risk: " + esc(p.validation.risk) + ")</p>";
      if (p.validation.reasons && p.validation.reasons.length) {
        html += "<ul>";
        p.validation.reasons.forEach(function (r) { html += "<li>" + esc(r) + "</li>"; });
        html += "</ul>";
      }
    }
    if (p.safety_notes) html += "<p><em>" + esc(p.safety_notes) + "</em></p>";
    if (p.rollback_notes) html += "<p>Rollback: " + esc(p.rollback_notes) + "</p>";
    if (p.reject_reason) html += "<p>Rejection: " + esc(p.reject_reason) + "</p>";
    detail.innerHTML = html;

    // Render diffs
    if (diffBox && p.files) {
      var dhtml = "";
      p.files.forEach(function (f) {
        dhtml += '<div class="diff-line diff-hdr">--- ' + esc(f.path) + ' (' + esc(f.action) + ')</div>';
        if (f.diff) {
          f.diff.split("\n").forEach(function (line) {
            var cls = "diff-ctx";
            if (line.startsWith("+")) cls = "diff-add";
            else if (line.startsWith("-")) cls = "diff-del";
            else if (line.startsWith("@@")) cls = "diff-hdr";
            dhtml += '<div class="diff-line ' + cls + '">' + esc(line) + '</div>';
          });
        } else {
          dhtml += '<div class="diff-line diff-ctx">(no diff)</div>';
        }
      });
      diffBox.innerHTML = dhtml;
    }

    // Action buttons
    if (actions) {
      var btns = "";
      if (p.status === "proposed" || p.status === "validated") {
        btns += '<button type="button" class="action-btn" id="btn-patch-validate">Validate</button> ';
      }
      if (p.status === "validated") {
        btns += '<button type="button" class="action-btn" id="btn-patch-apply" style="background:#238636">Apply</button> ';
      }
      if (p.status !== "applied" && p.status !== "rejected") {
        btns += '<button type="button" class="action-btn" id="btn-patch-reject" style="background:#da3633">Reject</button>';
      }
      actions.innerHTML = btns;

      var valBtn = $("#btn-patch-validate");
      if (valBtn) valBtn.addEventListener("click", function () { validatePatch(id); });
      var appBtn = $("#btn-patch-apply");
      if (appBtn) appBtn.addEventListener("click", function () { applyPatch(id); });
      var rejBtn = $("#btn-patch-reject");
      if (rejBtn) rejBtn.addEventListener("click", function () { rejectPatch(id); });
    }
  }

  async function validatePatch(id) {
    var detail = $("#patch-detail");
    var r = await api("POST", "/api/patches/" + id + "/validate");
    if (r.ok) {
      loadPatchDetail(id);
      loadPatches();
    } else {
      if (detail) detail.innerHTML += '<p class="error">Validation error: ' + esc(r.data.detail || "unknown") + '</p>';
    }
  }

  async function applyPatch(id) {
    var detail = $("#patch-detail");
    var r = await api("POST", "/api/patches/" + id + "/apply");
    if (r.ok) {
      loadPatchDetail(id);
      loadPatches();
    } else {
      if (detail) detail.innerHTML += '<p class="error">Apply error: ' + esc(r.data.detail || "unknown") + '</p>';
    }
  }

  async function rejectPatch(id) {
    var reason = prompt("Rejection reason (optional):");
    var r = await api("POST", "/api/patches/" + id + "/reject", { reason: reason || "" });
    if (r.ok) {
      loadPatchDetail(id);
      loadPatches();
    }
  }

  // Patch form
  (function () {
    var form = $("#patch-form");
    if (!form) return;
    form.addEventListener("submit", async function (e) {
      e.preventDefault();
      var title = $("#patch-title").value.trim();
      var desc = $("#patch-desc").value.trim();
      var path = $("#patch-path").value.trim();
      var action = $("#patch-action").value;
      var content = $("#patch-content").value;
      if (!path) { alert("File path is required"); return; }
      if (!content && action === "create") { alert("Content is required for create"); return; }
      var r = await api("POST", "/api/patches/propose", {
        title: title || "Untitled patch",
        description: desc,
        files: [{ path: path, action: action, after: content }]
      });
      if (r.ok) {
        form.reset();
        loadPatches();
        loadPatchDetail(r.data.id);
      } else {
        alert("Error: " + (r.data.detail || r.data.error || "unknown"));
      }
    });
  })();
})();
