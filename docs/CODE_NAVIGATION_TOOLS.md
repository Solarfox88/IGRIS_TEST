# Code Navigation Tools — Epic #59

## Overview

Provides safe, governed tools for the agent to see and understand a codebase.
All tools enforce path guard, secret guard, and output limits. Output is
structured for consumption by the Context Manager and Agent Reasoning Loop.

## Tools

### search_code(pattern, path?, max_results?, context_lines?)

Search for regex patterns in code files.

```json
POST /api/nav/search-code
{
  "pattern": "def create_app",
  "path": "igris/",
  "max_results": 50,
  "context_lines": 2
}
```

### find_files(pattern, max_results?)

Find files by name or glob pattern.

```json
POST /api/nav/find-files
{
  "pattern": "*.py",
  "max_results": 100
}
```

### list_directory(path, depth?, max_entries?)

List directory contents with optional recursion.

```json
POST /api/nav/list-directory
{
  "path": "igris/core",
  "depth": 2,
  "max_entries": 200
}
```

### read_file_range(path, start?, end?, max_lines?)

Read specific lines from a file.

```json
POST /api/nav/read-file-range
{
  "path": "igris/web/server.py",
  "start": 1,
  "end": 50
}
```

### repo_map()

Build a lightweight map of the repository.

```json
GET /api/nav/repo-map
```

### find_symbol(symbol, path?, max_results?)

Find symbol definitions (function, class, variable) by name.

```json
POST /api/nav/find-symbol
{
  "symbol": "create_app",
  "path": "igris/"
}
```

## Safety Guards

All tools enforce:

| Guard | Description |
|---|---|
| Path guard | All paths must be within project root |
| Secret guard | `.env`, keys, credentials are never read |
| Output limit | Results capped per query |
| Secret redaction | All output passes through `redact_secrets()` |
| Binary skip | Binary files (png, jpg, exe, etc.) are skipped |
| Dir skip | `.git`, `__pycache__`, `node_modules` are skipped |

## Connection to Agent Action Schema

The Code Navigation Tools map directly to action types from Epic #58:

| Action Type | Navigation Tool |
|---|---|
| `search_code` | `CodeNavigator.search_code()` |
| `find_files` | `CodeNavigator.find_files()` |
| `list_directory` | `CodeNavigator.list_directory()` |
| `read_file_range` | `CodeNavigator.read_file_range()` |

All are routed to `code_navigation` category and are read-only / no side effects.
