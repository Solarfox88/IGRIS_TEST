# Cost Routing

To keep operation affordable, IGRIS_GPT chooses between different language models depending on availability, cost and capability.

1. **Local provider (Ollama)** – The default provider is a locally hosted model such as `phi4-mini`.  Responses are fast and incur no API cost but the model quality may be lower.
2. **Fallback provider (OpenAI)** – When the local model cannot answer or the task requires higher accuracy, the agent falls back to a remote provider like OpenAI.  The API key must be supplied in the `.env` file.  The fallback model is only used when necessary.
3. **Vast.ai** – For compute‑intensive tasks or when a high‑end GPU model is required, Vast.ai instances can be spun up.  This is not implemented in the MVP but the scaffolding exists in the configuration.

The `/api/routing/explain` endpoint reports which provider was used for recent interactions and why.  Future versions will implement automatic cost estimation and budget enforcement.