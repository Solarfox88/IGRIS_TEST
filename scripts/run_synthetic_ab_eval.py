#!/usr/bin/env python3
"""
Synthetic A/B evaluation runner — Epic #445.

Sends fixture packets to both API helpers (Codex primary + DeepSeek shadow),
scores responses, writes records as source=synthetic_fixture.

Usage:
  python scripts/run_synthetic_ab_eval.py [--out .igris/helper_ab_results.json]
  python scripts/run_synthetic_ab_eval.py --dry-run   # score only, do not call APIs
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env so API keys are available to subprocess helper calls
_dotenv_path = ROOT / ".env"
if _dotenv_path.exists():
    for _line in _dotenv_path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _k = _k.strip()
        _v = _v.strip().strip('"').strip("'")
        if _k and _k not in os.environ:  # don't override already-set vars
            os.environ[_k] = _v

from igris.core.helper_ab_eval import (
    make_ab_record, save_ab_result, load_ab_results,
    score_helper_response, is_safe_to_switch,
)

FIXTURES_DIR = ROOT / "tests" / "fixtures" / "helper_eval"
DEFAULT_OUT  = str(ROOT / ".igris" / "helper_ab_results.json")
HELPER_CMD   = os.environ.get(
    "IGRIS_API_HELPER_COMMAND",
    f"{ROOT}/.venv/bin/python {ROOT}/scripts/igris_api_helper.py",
)
PRIMARY_MODEL = os.environ.get("IGRIS_API_HELPER_MODEL", "gpt-5.3-codex")
ALT_MODEL     = os.environ.get("IGRIS_API_HELPER_ALT_MODEL", "deepseek-v4-pro")
ALT_PROVIDER  = os.environ.get("IGRIS_API_HELPER_ALT_PROVIDER", "deepseek")
TIMEOUT       = int(os.environ.get("IGRIS_HELPER_TIMEOUT", "60"))


# ---------------------------------------------------------------------------
# Helper caller
# ---------------------------------------------------------------------------

def _call_helper(packet: dict, model: str, extra_env: dict | None = None) -> tuple[dict, float, int]:
    """Call igris_api_helper.py. Returns (parsed_response, latency_ms, returncode)."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    payload = json.dumps({"packet": packet, "model": model, "max_tokens": 1200})
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            HELPER_CMD.split(),
            input=payload, capture_output=True, text=True,
            timeout=TIMEOUT, env=env,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        try:
            parsed = json.loads(result.stdout) if result.stdout.strip() else {}
        except json.JSONDecodeError:
            parsed = {"_raw": result.stdout[:300]}
        return parsed, latency_ms, result.returncode
    except subprocess.TimeoutExpired:
        return {"_error": "timeout"}, TIMEOUT * 1000, 1
    except Exception as exc:
        return {"_error": str(exc)}, 0, 1


# ---------------------------------------------------------------------------
# Fixture → packet builder
# ---------------------------------------------------------------------------

def _fixture_to_packet(fix: dict) -> dict:
    return {
        "failure_class": fix.get("failure_class", "unknown"),
        "goal": fix.get("goal", ""),
        "recent_events": fix.get("recent_events", []),
        "diff_summary": fix.get("diff_summary", ""),
        "test_output": fix.get("test_output", ""),
        "context": "synthetic_fixture_evaluation",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(out_path: str, dry_run: bool) -> int:
    fixtures = sorted(FIXTURES_DIR.glob("*.json"))
    if not fixtures:
        print(f"ERROR: no fixtures found in {FIXTURES_DIR}", file=sys.stderr)
        return 2

    # Load existing records to avoid duplicate synthetic runs
    existing = load_ab_results(out_path)
    done_cases = {r["case_id"] for r in existing if r.get("source") == "synthetic_fixture"}

    rows = []
    errors = 0

    print(f"Running synthetic A/B eval — {len(fixtures)} fixtures")
    print(f"Primary : {PRIMARY_MODEL}")
    print(f"Candidate: {ALT_MODEL}")
    print(f"Output  : {out_path}")
    if dry_run:
        print("DRY RUN — APIs not called, scores are placeholder 0.0")
    print()

    for fix_path in fixtures:
        fix = json.loads(fix_path.read_text())
        case_id = fix.get("case_id", fix_path.stem)
        skip_label = " (already done, skipping API)" if case_id in done_cases else ""

        if dry_run or case_id in done_cases:
            # Produce a zero-cost placeholder record
            zero_bd = {k: 0.0 for k in ["schema_valid","diagnosis_specificity",
                                          "execution_plan_actionability","acceptance_matrix_quality",
                                          "safety_compliance","no_secrets","decomposition_quality"]}
            record = make_ab_record(
                case_id=case_id,
                primary_model=PRIMARY_MODEL,
                alt_model=ALT_MODEL,
                primary_score=0.0, alt_score=0.0,
                primary_breakdown=zero_bd, alt_breakdown=zero_bd,
                primary_cost_usd=0.0, alt_cost_usd=0.0,
                source="synthetic_fixture",
            )
            record["_dry_run"] = True
            if dry_run:
                rows.append((case_id, 0.0, 0.0, 0.0, 0.0, 0, 0, "tie", "dry-run"))
                print(f"  {case_id:55s} [DRY RUN]")
                continue
            else:
                rows.append((case_id, 0.0, 0.0, 0.0, 0.0, 0, 0, "tie", f"skipped{skip_label}"))
                print(f"  {case_id:55s} [SKIP — already recorded]")
                continue

        packet = _fixture_to_packet(fix)

        # Call primary (Codex)
        print(f"  {case_id:55s} calling primary...", end="", flush=True)
        primary_resp, primary_latency, primary_rc = _call_helper(packet, PRIMARY_MODEL)
        print(f" {primary_latency}ms", end="", flush=True)

        # Call alt (DeepSeek) — clear IGRIS_API_HELPER_MODEL so _resolve_model
        # doesn't forward the Codex model name to the DeepSeek endpoint.
        print(f"  → candidate...", end="", flush=True)
        alt_resp, alt_latency, alt_rc = _call_helper(
            packet, ALT_MODEL,
            extra_env={
                "IGRIS_API_HELPER_MODE": "auto",
                "IGRIS_API_HELPER_PROVIDER": ALT_PROVIDER,
                "IGRIS_API_HELPER_MODEL": ALT_MODEL,
                "IGRIS_HELPER_AB_ARM": "alt",
            },
        )
        print(f" {alt_latency}ms", end="", flush=True)

        # Score
        case_ctx = {"expected_good_response_traits": fix.get("expected_good_response_traits", {}),
                    "failure_class": fix.get("failure_class", "")}
        primary_score_r = score_helper_response(primary_resp, case_ctx)
        alt_score_r     = score_helper_response(alt_resp, case_ctx)

        primary_cost = float(primary_resp.get("estimated_cost_usd", 0.0))
        alt_cost     = float(alt_resp.get("estimated_cost_usd", 0.0))

        record = make_ab_record(
            case_id=case_id,
            primary_model=PRIMARY_MODEL,
            alt_model=ALT_MODEL,
            primary_score=primary_score_r["total"],
            alt_score=alt_score_r["total"],
            primary_breakdown=primary_score_r["breakdown"],
            alt_breakdown=alt_score_r["breakdown"],
            primary_cost_usd=primary_cost,
            alt_cost_usd=alt_cost,
            primary_latency_ms=primary_latency,
            alt_latency_ms=alt_latency,
            source="synthetic_fixture",
        )

        if primary_rc != 0 or alt_rc != 0:
            errors += 1
            record["_warnings"] = {
                "primary_rc": primary_rc,
                "alt_rc": alt_rc,
                "primary_issues": primary_score_r["issues"],
                "alt_issues": alt_score_r["issues"],
            }

        save_ab_result(record, out_path)

        winner = record["winner"]
        rows.append((
            case_id,
            primary_score_r["total"], alt_score_r["total"],
            primary_cost, alt_cost,
            primary_latency, alt_latency,
            winner,
            ", ".join(primary_score_r["issues"] + alt_score_r["issues"]) or "ok",
        ))
        print(f"  primary={primary_score_r['total']:.2f} alt={alt_score_r['total']:.2f} [{winner}]")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 110)
    hdr = f"{'case_id':40s} {'p_score':>7} {'a_score':>7} {'p_cost':>8} {'a_cost':>8} {'p_ms':>6} {'a_ms':>6} {'winner':>8}  notes"
    print(hdr)
    print("-" * 110)
    for row in rows:
        case_id, ps, as_, pc, ac, pm, am, winner, notes = row
        print(f"{case_id:40s} {ps:7.3f} {as_:7.3f} {pc:8.5f} {ac:8.5f} {pm:6d} {am:6d} {winner:>8}  {notes[:30]}")
    print("=" * 110)

    # Switch policy report
    all_records = load_ab_results(out_path)
    report = is_safe_to_switch(all_records)
    print()
    print("Switch policy report:")
    print(f"  synthetic_count       : {report['synthetic_count']}")
    print(f"  organic_count         : {report['organic_count']}")
    print(f"  failure_classes       : {report['failure_classes_covered']}")
    print(f"  safe_to_switch        : {report['safe_to_switch']}")
    if not report["safe_to_switch"]:
        print(f"  reason_if_not_safe    : {report['reason_if_not_safe']}")
    for reason in report["reasons"]:
        mark = "✓" if "✓" in reason else "✗" if any(w in reason for w in ("need","failure","cost","regress","<")) else " "
        print(f"    [{mark}] {reason}")
    print()

    return 2 if errors and not dry_run else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Synthetic A/B helper evaluation")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true", help="Skip API calls, just show table structure")
    ap.add_argument("--primary", default=None, help="Override primary model")
    ap.add_argument("--candidate", default=None, help="Override candidate model")
    ap.add_argument("--fixtures", default=None, help="Override fixtures directory")
    args = ap.parse_args()

    if args.primary:
        PRIMARY_MODEL = args.primary
    if args.candidate:
        ALT_MODEL = args.candidate
    if args.fixtures:
        FIXTURES_DIR = Path(args.fixtures)

    sys.exit(run(args.out, args.dry_run))
