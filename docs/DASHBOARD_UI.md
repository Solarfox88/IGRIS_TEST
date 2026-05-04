# Dashboard UI — Tab Simplification and Mission Control

**Sprint 34 — v0.6 Human-Usable Console**

## Overview

Reduced 14 tabs to 7 primary tabs with sub-tab navigation for grouped functionality. Enhanced Mission Control into a proper Dashboard with health/readiness/diagnostics/loop summary cards.

## Tab Layout

### Before (14 tabs)
Mission | Terminal | Files | Git | Tests | Logs | Agent | Tasks | Safety | Cost | A2A | Memory | Loop | Patches

### After (7 tabs)
Dashboard | Code | Tasks | Terminal | Memory | Safety | Advanced

### Grouping

| Primary Tab | Sub-tabs | Contains |
|------------|----------|----------|
| **Dashboard** | — | Health, Readiness, Diagnostics, Loop Status, Missions, Decision Reports |
| **Code** | Files, Git, Patches | File browser, git status/diff/branches/commit/PR, patch proposals |
| **Tasks** | Tasks, Loop | Task CRUD, teacher remediation, autonomous loop |
| **Terminal** | Commands, Tests | Safe terminal commands, test runner |
| **Memory** | Memory, Timeline | Decision/failure memory, agent timeline |
| **Safety** | Safety, Cost & Routing | Safety status, execution reports, cost/budget/routing |
| **Advanced** | A2A, Logs | A2A protocol, system logs |

## Dashboard Cards

The dashboard displays a 2x2 grid of summary cards:

1. **System Health** — Server health status
2. **Readiness** — Provider, model, fallback availability
3. **Diagnostics** — Starvation, blocked tasks, family health issues
4. **Loop Status** — Total steps, last action

Below the cards: project context, missions, new mission form, mission detail/graph, and recent decision reports.

## Sub-tab Navigation

Each grouped tab has a secondary navigation bar with underline-style active indicator. Clicking a sub-tab shows its content while hiding siblings. No page reload needed.

## API

### GET /api/dashboard/summary
Aggregated view returning health, readiness, diagnostics, loop status, and tab_layout metadata in a single call.

## Backward Compatibility

All original HTML element IDs are preserved (e.g., `mission-health`, `git-info`, `test-output`, etc.). JavaScript that references these IDs continues to work.

## Responsive

- Dashboard grid: 2 columns on desktop, 1 column on mobile (768px)
- Sub-tabs: smaller padding on mobile
- All existing responsive rules preserved
