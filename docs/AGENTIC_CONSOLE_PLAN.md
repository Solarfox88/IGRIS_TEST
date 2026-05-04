# Agentic Console Plan

The agentic console is the primary user interface for interacting with IGRIS.  It combines a chat experience with panels exposing project state, tests, file browser and logs.  The plan is to iterate on the interface in stages.  The initial version includes:

* **Mission Control** – the central chat window where you converse with the agent.  Messages are sent to the backend and responses are streamed back.  The session API supports creating new sessions and sending messages.
* **Files** – a read‑only file tree and preview panel.  Path traversal attacks are prevented and only text files under the project root are accessible.  Large files are truncated.
* **Git** – a read‑only view of the current branch, dirty state and recent commit history.  Pushing or committing through the UI is deferred until safety policies are defined.
* **Tests** – a button to run `pytest` with a timeout and show the results.  Only the pre‑configured test command is executed; arbitrary commands are not allowed.
* **Logs** – tail of application logs.  The API truncates log output to prevent memory exhaustion.
* **Safety** – displays the current set of allowed and blocked operations.  In future versions this panel will allow adjusting safety thresholds.
* **Cost** – explains which provider (local, OpenAI or Vast.ai) was used for recent responses and why.

Future iterations will add:

* A controlled terminal with a set of pre‑approved commands.
* A timeline of agent steps showing plans, actions, observations and corrective measures.
* Integration with Git for opening pull requests once the autonomy layer matures.