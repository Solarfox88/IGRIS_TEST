# Safe Terminal MVP Plan

Executing arbitrary shell commands from a web application is risky.  The safe terminal concept restricts command execution to a vetted list and enforces timeouts and resource limits.  The MVP includes:

* **Predefined commands** – Only a small set of safe commands are exposed, such as:
  - `git status --short` – show a concise status of the repository.
  - `git log --oneline -10` – show the last 10 commits.
  - `python -m pytest -q` – run the test suite quietly.
  - `ls` and `find` limited to the workspace root.
* **No arbitrary arguments** – Users cannot enter free‑form commands; instead they select from a menu and the server executes the corresponding command.
* **Timeout and output limits** – Each command is run with a maximum duration (e.g. 30 seconds).  Output is truncated to a reasonable length to avoid overwhelming the client.
* **Isolation** – Commands run in a restricted working directory (the project root) and cannot access other parts of the filesystem.
* **Logging** – All command executions are logged for auditability.

Future work includes allowing users to propose new commands that go through a review process, stronger sandboxing (e.g. via containers) and integration with the task engine for autonomous execution.