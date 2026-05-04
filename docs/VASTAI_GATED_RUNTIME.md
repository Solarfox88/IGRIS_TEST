# Vast.ai Gated DeepSeek Runtime

IGRIS_GPT provides a gated Vast.ai integration for GPU-accelerated LLM inference with DeepSeek R1.

## Safety Principles

1. **No auto-provisioning** — `VASTAI_AUTO_PROVISION=false` by default
2. **Approval required** — all destructive operations need `I_APPROVE_VASTAI_COSTS`
3. **No real API calls in CI** — all operations are mock/dry-run
4. **Budget gate** — cost must be within `VASTAI_MAX_HOURLY_COST`
5. **Anti-duplicate** — cannot provision if instance already active
6. **State-aware destroy** — cannot destroy non-existent instance
7. **No loop provisioning** — autonomous loop cannot provision instances

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VASTAI_API_KEY` | (none) | Vast.ai API key |
| `VASTAI_MODEL` | `deepseek-r1:32b` | Primary model |
| `VASTAI_FALLBACK_MODEL` | `qwen2.5-coder:7b` | Fallback model |
| `VASTAI_AUTO_PROVISION` | `false` | Auto-provision (always false) |
| `VASTAI_REQUIRE_APPROVAL` | `true` | Require approval token |
| `VASTAI_MAX_HOURLY_COST` | `0.50` | Maximum hourly cost in USD |
| `VASTAI_MODE` | `on_demand` | Mode: on_demand, always_on, disabled |

## Endpoints

### GET /api/vastai/config

Returns configuration (no API key exposed).

### GET /api/vastai/status

Returns instance status, mode, model.

### POST /api/vastai/estimate

Estimate cost for a model.

```json
{"model": "deepseek-r1:32b", "hours": 2.0}
```

### POST /api/vastai/offers/search

Search GPU offers (mock).

```json
{"model": "deepseek-r1:32b", "max_cost": 0.40}
```

### POST /api/vastai/provision (gated)

Provision a GPU instance. Requires approval.

```json
{"approval": "I_APPROVE_VASTAI_COSTS", "model": "deepseek-r1:32b"}
```

### POST /api/vastai/destroy (gated)

Destroy active instance. Requires approval.

```json
{"approval": "I_APPROVE_VASTAI_COSTS"}
```

### POST /api/vastai/set-mode (gated)

Change operating mode. Requires approval.

```json
{"mode": "disabled", "approval": "I_APPROVE_VASTAI_COSTS"}
```

## Supported Models

| Model | VRAM | Min GPU | Est. Cost/hr |
|-------|------|---------|-------------|
| deepseek-r1:32b | 24 GB | RTX 3090 | $0.30 |
| deepseek-r1:70b | 48 GB | A6000 | $0.60 |
| qwen2.5-coder:7b | 8 GB | RTX 3060 | $0.10 |
| qwen2.5-coder:32b | 24 GB | RTX 3090 | $0.30 |

## Real Cost Warning

When Vast.ai is connected with a real API key:
- Provisioning creates actual GPU instances with real costs
- Costs are billed per hour by Vast.ai
- Always verify budget before provisioning
- Destroy instances when not in use
- Monitor usage via `GET /api/vastai/status`

## Current Status

**Mock/Gated** — no real API calls are made. All operations return mock results.
To connect to real Vast.ai, set `VASTAI_API_KEY` and implement real HTTP calls in `vastai_manager.py`.
