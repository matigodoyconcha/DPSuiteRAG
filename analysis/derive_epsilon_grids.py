"""Turn workload budgets into each method's own knob.

The seven methods are compared at equal ``epsilon_total(Q)`` — the privacy loss over a
Q-question workload at the common delta — but none of them takes that number as input.
Each has its own knob in its own units: a per-token budget, a per-query budget, a
clipping bound. This inverts the shared accountant once per target and prints the value
to put in each config, so the grids are derived from the axis rather than chosen and
then relabelled.

Only the budget knob moves. Every other hyperparameter stays at the value its paper
fixed, so a point on the curve differs from its neighbour in the budget alone and not
in a re-tuning that happens to come with it.

Usage:
    python analysis/derive_epsilon_grids.py --targets 1 3 10 30 100 --queries 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.privacy_accounting_zcdp import calibrate_dp_synrag  # noqa: E402
from metrics.workload_accounting import (  # noqa: E402
    DEFAULT_DELTA,
    epsilon_query_for_workload,
    exponential_mechanism,
)

# Structural parameters, held at their papers' values while the budget moves.
DPVOTE_TOKENS = 50
SPARSE_RELEASES = 32
DPRAG_TOKENS = 70
DELTA_TOKEN = 1e-6
# DP-RAG splits its budget between retrieval and generation; the ratio is the paper's
# operating point (eps_r = 0.5 against a per-token generation budget), kept fixed so
# the sweep moves one scalar.
DPRAG_RETRIEVAL_RATIO = 0.5
SYNRAG_TABLE7 = dict(
    K=10, sigma_h=7.071, L=5, epsilon_theta_s=0.4, sigma_mu=7.454, T=70, c=0.1057, tau=1.0
)


def derive(targets: list[float], queries: int, delta: float) -> dict:
    grids: dict[str, dict] = {}

    def invert(build, label):
        out = {}
        for target in targets:
            try:
                out[str(target)] = round(
                    epsilon_query_for_workload(
                        build, target_epsilon=target, queries=queries, delta=delta
                    ),
                    6,
                )
            except (ValueError, RuntimeError) as exc:
                # A target the mechanism cannot reach is recorded, not silently dropped:
                # a missing point on a curve should be visible as a refusal.
                out[str(target)] = {"unreachable": str(exc)}
        grids[label] = out

    invert(
        lambda e: [exponential_mechanism("vote", e, count=DPVOTE_TOKENS, delta_add=DELTA_TOKEN)],
        "dpvoterag.epsilon_token",
    )
    invert(
        lambda e: [
            exponential_mechanism("vote", e, count=SPARSE_RELEASES, delta_add=DELTA_TOKEN)
        ],
        "dpsparsevoterag.epsilon_token",
    )
    invert(
        lambda e: [
            exponential_mechanism("retrieval", DPRAG_RETRIEVAL_RATIO * e),
            exponential_mechanism("generation", e, count=DPRAG_TOKENS),
        ],
        "dp_rag.token_epsilon",
    )

    # Flat in the workload: the per-document filter charges M * eps_q regardless of how
    # long the stream runs, so the knob is the target itself.
    grids["private_rag.epsilon_q (M=1)"] = {str(t): float(t) for t in targets}

    # Also flat, and closed-form: one offline build covers the whole workload.
    synrag = {}
    for target in targets:
        try:
            calibrated = calibrate_dp_synrag(
                **SYNRAG_TABLE7, target_epsilon=target, delta=delta
            )
            synrag[str(target)] = {
                key: round(calibrated[key], 6)
                for key in ("c", "sigma_h", "sigma_mu", "epsilon_theta_s")
            }
        except ValueError as exc:
            synrag[str(target)] = {"unreachable": str(exc)}
    grids["dp_synrag (all four stages)"] = synrag

    # DP-KSA calibrates from a per-query (eps, delta) pair, so the shared axis gives it
    # the per-query epsilon and its own routine does the rest.
    invert(lambda e: [exponential_mechanism("query", e)], "dpksa.epsilon (per query)")
    return grids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", type=float, default=[1, 3, 10, 30, 100])
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--output", type=str, default="results/epsilon_grids.json")
    args = parser.parse_args()

    grids = derive(args.targets, args.queries, args.delta)
    print(f"epsilon_total targets {args.targets} over Q={args.queries} at delta={args.delta:g}\n")
    for label, values in grids.items():
        print(f"{label}:")
        for target, value in values.items():
            print(f"    eps_total={target:>6}  ->  {value}")
        print()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"queries": args.queries, "delta": args.delta, "grids": grids}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
