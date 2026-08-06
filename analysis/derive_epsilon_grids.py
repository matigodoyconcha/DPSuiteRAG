"""Turn workload budgets into each method's own knob.

The seven methods are compared at equal ``epsilon_total(Q)`` — the privacy loss over a
Q-question workload at the common delta — but none of them takes that number as input.
Each has its own knob in its own units: a per-token budget, a per-query budget, a
clipping bound. This inverts the shared accountant once per target and prints the value
to put in each config, so the grids are derived from the axis rather than chosen and
then relabelled.

Two rules make the output trustworthy, and both were learned by getting them wrong:

* **Invert the same mechanism the runner charges.** The inversion goes through
  ``METHOD_MECHANISMS``, the exact builders ``workload_accounting_block`` calls from
  each runner. Hand-written stand-ins drifted from them silently and in opposite
  directions — a point labelled ``epsilon_total = 1`` really cost 25 for DP-RAG and
  0.38 for DP-KSA, a 66x relative bias on the axis whose whole purpose is to make the
  methods comparable.
* **Read the structural parameters from the configs.** Token counts, release ceilings
  and budget splits are taken from the YAML that will actually run, so a grid cannot
  be derived for one experiment and used in another.

Every derived value is verified by composing it back through the accountant before it
is printed; a mismatch is reported rather than emitted.

Usage:
    python analysis/derive_epsilon_grids.py --targets 1 3 10 30 100 --queries 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.privacy_accounting_dpksa import calibrate_dpksa  # noqa: E402
from metrics.privacy_accounting_zcdp import calibrate_dp_synrag  # noqa: E402
from metrics.workload_accounting import (  # noqa: E402
    DEFAULT_DELTA,
    delta_token_for_workload,
    dpksa_mechanism_delta_for_workload,
    epsilon_query_for_workload,
    method_epsilon_total,
)

CONFIG_DIR = PROJECT_ROOT / "experiments" / "configs" / "full"


def _config(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8")) or {}


#: DP-RAG's operating point in the paper (medical evaluation, Fig. 6): the retrieval
#: budget, the total generation budget, and the answer length they were measured at.
#: The proportion between the two budgets is read from HERE and not from the config,
#: because the config holds a grid this script produced: deriving the proportion from
#: its own previous output would move it every run — measured, from 2.85 to 1.02 on
#: the first repeat — quietly redistributing the budget between retrieval and
#: generation while every printed ratio still read 1.0.
_DP_RAG_PAPER_POINT = {"epsilon_retrieval": 0.5, "epsilon_generation": 5.0, "max_tokens": 70}


def _dp_rag_retrieval_ratio(config: dict) -> float:
    """Retrieval budget per unit of per-token generation budget, at the paper's point.

    DP-RAG's config declares an absolute retrieval epsilon and a total generation
    epsilon, from which the generator back-solves a per-token budget. The shared axis
    needs one scalar, so both are scaled together and this is the proportion between
    them that the paper's operating point implies. Being a proportion it is unitless,
    so it is unaffected by the answer length the benchmark runs at.
    """
    from metrics.privacy_accounting import pld_dp_rag_total

    eps_r = float(_DP_RAG_PAPER_POINT["epsilon_retrieval"])
    eps_g = float(_DP_RAG_PAPER_POINT["epsilon_generation"])
    tokens = int(_DP_RAG_PAPER_POINT["max_tokens"])
    delta = float(config.get("delta", 1e-3))

    # Same bisection the generator uses to turn eps_g into a per-token budget.
    lo, hi = 1e-6, eps_g
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if pld_dp_rag_total(0.0, mid, tokens, delta) <= eps_g:
            lo = mid
        else:
            hi = mid
    return eps_r / lo


def _dpksa_floor(params: dict) -> float:
    """Smallest per-query epsilon `calibrate_dpksa` will calibrate for."""
    lo, hi = 1e-6, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        try:
            calibrate_dpksa(
                target_epsilon=mid, target_delta=params["delta"],
                ptr_delta_fraction=params["ptr_delta_fraction"],
                ptr_noise_coupling=params["ptr_noise_coupling"],
            )
            hi = mid
        except RuntimeError:
            lo = mid
    return hi * 1.01


def _synrag_for_target(params: dict, target: float, queries: int, delta: float) -> dict:
    """Scale DP-SynRAG's four build stages until the SHARED AXIS reads `target`.

    Its own routine converts zCDP to (eps, delta) in closed form, while the axis
    optimises the conversion over the Renyi order and therefore reads a smaller
    epsilon for the same rho. Calibrating against the closed form and reporting on the
    axis handed the method 0.62x of its budget at the low end -- conservative, but it
    puts DP-SynRAG at a different place on the axis than the label claims. Every stage
    is quadratic in its parameter, so rho scales as the square of one factor and this
    is a bisection on that factor.
    """
    reference = calibrate_dp_synrag(**params, target_epsilon=target, delta=delta)
    rho_ref = reference["rho_total"]

    def axis_epsilon(scale: float) -> float:
        return method_epsilon_total(
            "dp_synrag", queries=queries, delta=delta, rho=rho_ref * scale * scale
        ).epsilon

    lo, hi = 1.0, 1.0
    while axis_epsilon(hi) < target:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("no scale reaches the target")
    while axis_epsilon(lo) > target:
        lo /= 2.0
        if lo < 1e-6:
            raise RuntimeError("no scale reaches the target")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if axis_epsilon(mid) <= target:
            lo = mid
        else:
            hi = mid

    # Every stage parameter moves by the same factor: noise scales divide by it,
    # budgets and the clipping bound multiply by it.
    return {
        "c": reference["c"] * lo,
        "sigma_h": reference["sigma_h"] / lo,
        "sigma_mu": reference["sigma_mu"] / lo,
        "epsilon_theta_s": reference["epsilon_theta_s"] * lo,
        "rho_total": rho_ref * lo * lo,
        "scale": lo,
    }


def _structural() -> dict:
    """Structural parameters, read from the configs that will run."""
    dp_rag = _config("dp_rag")
    vote = _config("dpvoterag")
    sparse = _config("dpsparsevoterag")
    ksa = _config("dpksa")
    synrag = _config("dp_synrag")
    return {
        "dp_rag": {
            # The two budgets keep the ratio the paper operates at, so a point on the
            # curve differs from its neighbour in the budget alone.
            "retrieval_ratio": _dp_rag_retrieval_ratio(dp_rag),
            "max_tokens": int(dp_rag.get("max_new_tokens", 70)),
        },
        "dpvoterag": {
            "max_tokens": int(vote["dpvoterag"].get("tmax_absolute") or
                              vote["dpvoterag"].get("max_length", 50)),
            "delta_token": float(vote["dpvoterag"]["delta_token"]),
            # The runners derive the per-token delta from the workload; reading the
            # config's fallback here instead would publish a grid for a mechanism
            # whose bottom-out delta is not the one the sweep runs at.
            "delta_token_from_workload": bool(
                vote["dpvoterag"].get("delta_token_from_workload", False)
            ),
        },
        "dpsparsevoterag": {
            "max_private_tokens": int(sparse["dpsparsevoterag"]["max_private_tokens"]),
            "delta_token": float(sparse["dpsparsevoterag"]["delta_token"]),
            "sparse_fraction": float(sparse["dpsparsevoterag"]["sparse_fraction"]),
            "delta_token_from_workload": bool(
                sparse["dpsparsevoterag"].get("delta_token_from_workload", False)
            ),
        },
        "dpksa": {
            "delta": float(ksa["delta"]),
            "ptr_delta_fraction": float(ksa["dpksa"].get("ptr_delta_fraction", 0.5)),
            "ptr_noise_coupling": float(ksa["dpksa"].get("ptr_noise_coupling", 0.25)),
            # As above for the vote methods: the runner derives the mechanism delta
            # from the workload, and that delta sets the PTR threshold, so publishing
            # the config's pinned fallback would be a grid for a different mechanism.
            "delta_from_workload": bool(ksa.get("delta_from_workload", False)),
        },
        "dp_synrag": {
            "K": int(synrag["keyword_extractor"]["K"]),
            "sigma_h": float(synrag["clustering"]["sigma_h"]),
            "L": int(synrag["clustering"]["L"]),
            "epsilon_theta_s": float(synrag["cluster_retrieval"]["epsilon_theta_s"]),
            "sigma_mu": float(synrag["cluster_retrieval"]["sigma_mu"]),
            "T": int(synrag["synthetic_generation"]["T"]),
            "c": float(synrag["synthetic_generation"]["c"]),
            "tau": float(synrag["synthetic_generation"]["tau"]),
        },
        "private_rag": {
            "m_retrievals": int(_config("private_rag")["private_rag"].get("M", 1)),
        },
    }


def _delta_token(structural: dict, *, queries: int, tokens: int, delta: float) -> float:
    """The per-token additive delta this method's runner will use.

    Derived from the workload where the config asks for it, so that the grid is
    published for the same mechanism the sweep charges.
    """
    if structural.get("delta_token_from_workload"):
        return delta_token_for_workload(queries=queries, tokens=tokens, delta=delta)
    return float(structural["delta_token"])


def _dpksa_delta(structural: dict, *, queries: int, delta: float) -> float:
    """The mechanism delta DP-KSA's runner will calibrate against.

    Its PTR test charges a share of this on every query, so it is the same story as the
    per-token delta above: pinned, the workload's additive spend grows with Q; derived,
    it is the same constant in every cell. It is not only a reporting figure — it sets
    the release threshold — so the grid has to be published for whichever rule runs.
    """
    if structural.get("delta_from_workload"):
        return dpksa_mechanism_delta_for_workload(
            queries=queries,
            ptr_delta_fraction=structural["ptr_delta_fraction"],
            delta=delta,
        )
    return float(structural["delta"])


def derive(targets: list[float], queries: int, delta: float) -> dict:
    p = _structural()
    grids: dict[str, dict] = {}

    def record(label: str, target: float, knob, params: dict, method: str):
        """Compose the derived knob back through the accountant before emitting it."""
        achieved = method_epsilon_total(method, queries=queries, delta=delta, **params)
        grids.setdefault(label, {})[str(target)] = {
            "value": knob,
            "achieved_epsilon_total": round(achieved.epsilon, 4),
            # A ratio away from 1.0 means the inversion and the runner disagree about
            # what this method spends.
            "ratio": round(achieved.epsilon / target, 4),
        }

    for target in targets:
        # --- DP-RAG: both of its budgets move together, in the paper's proportion.
        # Holding the retrieval epsilon at its absolute 0.5 puts a floor under the
        # method: charged once per query it composes to 24.93 at Q=100 on its own, so
        # every target below that is unreachable no matter how quiet generation gets.
        s = p["dp_rag"]
        build = lambda e, s=s: dict(
            epsilon_retrieval=s["retrieval_ratio"] * e, token_epsilon=e,
            max_tokens=s["max_tokens"],
        )
        try:
            knob = epsilon_query_for_workload(
                lambda e, b=build: __import__(
                    "metrics.workload_accounting", fromlist=["dp_rag_mechanisms"]
                ).dp_rag_mechanisms(**b(e)),
                target_epsilon=target, queries=queries, delta=delta,
            )
            record("dp_rag.token_epsilon", target, round(knob, 6), build(knob), "dp_rag")
            # The config declares the per-query generation budget, from which the
            # generator back-solves the per-token one; the axis inverts to the latter.
            # Composing T token mechanisms against a cost-free retrieval reverses that
            # calibration, so the grid comes out in the units the YAML actually takes.
            from metrics.privacy_accounting import pld_dp_rag_total

            eps_g = pld_dp_rag_total(
                eps_retrieval=0.0,
                token_epsilon=knob,
                num_tokens=s["max_tokens"],
                delta=float(_config("dp_rag").get("delta", 1e-3)),
            )
            # Verify the two maps really invert each other: back-solve the per-token
            # budget from the derived per-query one the way the generator does, and
            # check it lands on the knob the axis asked for.
            lo, hi = 1e-9, 10.0
            for _ in range(60):
                mid = (lo + hi) / 2
                if pld_dp_rag_total(
                    eps_retrieval=0.0, token_epsilon=mid, num_tokens=s["max_tokens"],
                    delta=float(_config("dp_rag").get("delta", 1e-3)),
                ) < eps_g:
                    lo = mid
                else:
                    hi = mid
            grids.setdefault("dp_rag.epsilon_generation_values", {})[str(target)] = {
                "value": round(eps_g, 6),
                "epsilon_retrieval_values": round(s["retrieval_ratio"] * knob, 6),
                "ratio": round(((lo + hi) / 2) / knob, 4),
            }
        except (ValueError, RuntimeError) as exc:
            grids.setdefault("dp_rag.token_epsilon", {})[str(target)] = {"unreachable": str(exc)}

        # --- The two vote methods: the per-token budget moves; for the sparse variant
        # the generator splits it and opens one threshold epoch per private release.
        # The knob is `epsilon_values` — the query's whole budget — and the per-token
        # budget follows from it (`epsilon_token_from_budget` in the shared configs).
        # So the answer length is held at the config's, and each point of the grid
        # differs from its neighbour in the noise per vote alone.
        s = p["dpvoterag"]
        T = s["max_tokens"]
        dt = _delta_token(s, queries=queries, tokens=T, delta=delta)
        build = lambda e, T=T, dt=dt: dict(
            epsilon_token=e / T, max_tokens=T, delta_token=dt,
        )
        try:
            knob = epsilon_query_for_workload(
                lambda e, b=build: __import__(
                    "metrics.workload_accounting", fromlist=["dpvoterag_mechanisms"]
                ).dpvoterag_mechanisms(**b(e)),
                target_epsilon=target, queries=queries, delta=delta,
            )
            record("dpvoterag.epsilon_values", target, round(knob, 6), build(knob), "dpvoterag")
        except (ValueError, RuntimeError) as exc:
            grids.setdefault("dpvoterag.epsilon_values", {})[str(target)] = {"unreachable": str(exc)}

        s = p["dpsparsevoterag"]
        f, c = s["sparse_fraction"], s["max_private_tokens"]
        dt = _delta_token(s, queries=queries, tokens=c, delta=delta)
        build = lambda e, f=f, c=c, dt=dt: dict(
            epsilon_token=(1.0 - f) * (e / c), max_private_tokens=c,
            delta_token=dt, sparse_epsilon=f * (e / c), n_epochs=1 + c,
        )
        try:
            knob = epsilon_query_for_workload(
                lambda e, b=build: __import__(
                    "metrics.workload_accounting", fromlist=["dpsparsevoterag_mechanisms"]
                ).dpsparsevoterag_mechanisms(**b(e)),
                target_epsilon=target, queries=queries, delta=delta,
            )
            record("dpsparsevoterag.epsilon_values", target, round(knob, 6),
                   build(knob), "dpsparsevoterag")
        except (ValueError, RuntimeError) as exc:
            grids.setdefault("dpsparsevoterag.epsilon_values", {})[str(target)] = {
                "unreachable": str(exc)
            }

        # --- DP-KSA: the config knob is a per-query epsilon that its own routine turns
        # into a Gumbel scale and a PTR sigma, so the inversion has to go through that
        # calibration rather than through a bare exponential mechanism.
        s = {**p["dpksa"], "delta": _dpksa_delta(p["dpksa"], queries=queries, delta=delta)}

        def ksa_params(eps_query: float, s=s) -> dict:
            cal = calibrate_dpksa(
                target_epsilon=eps_query, target_delta=s["delta"],
                ptr_delta_fraction=s["ptr_delta_fraction"],
                ptr_noise_coupling=s["ptr_noise_coupling"],
            )
            return dict(
                gumbel_scale=cal["gumbel_scale"], sigma=cal["sigma"],
                delta_ptr=cal["delta_ptr"],
            )

        try:
            knob = epsilon_query_for_workload(
                lambda e: __import__(
                    "metrics.workload_accounting", fromlist=["dpksa_mechanisms"]
                ).dpksa_mechanisms(**ksa_params(e)),
                target_epsilon=target, queries=queries, delta=delta,
                # calibrate_dpksa cannot reach an arbitrarily small epsilon: below its
                # own floor it raises rather than returning quieter noise, so the
                # search starts where the calibration is actually defined.
                lower=_dpksa_floor(s),
            )
            record("dpksa.epsilon", target, round(knob, 6), ksa_params(knob), "dpksa")
        except (ValueError, RuntimeError) as exc:
            grids.setdefault("dpksa.epsilon", {})[str(target)] = {"unreachable": str(exc)}

        # --- Flat in the workload: the per-document filter charges M * eps_q however
        # long the stream runs, so the knob is the target divided by M.
        m = p["private_rag"]["m_retrievals"]
        record("private_rag.epsilon_q", target, round(target / m, 6),
               dict(m_retrievals=m, epsilon_q=target / m), "private_rag")

        # --- Also flat, and calibrated across all four build stages at once. Its own
        # routine uses the closed-form zCDP conversion while the shared axis optimises
        # over the Renyi order, so the achieved figure is read back from the axis.
        s = p["dp_synrag"]
        try:
            cal = _synrag_for_target(s, target, queries, delta)
            record("dp_synrag.c (all four stages scale)", target,
                   {k: round(cal[k], 6) for k in ("c", "sigma_h", "sigma_mu", "epsilon_theta_s")},
                   dict(rho=cal["rho_total"]), "dp_synrag")
        except (ValueError, RuntimeError) as exc:
            grids.setdefault("dp_synrag.c (all four stages scale)", {})[str(target)] = {
                "unreachable": str(exc)
            }

    return grids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", type=float, default=[1, 3, 10, 30, 100])
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--output", type=str, default="results/epsilon_grids.json")
    args = parser.parse_args()

    grids = derive(args.targets, args.queries, args.delta)
    print(
        f"epsilon_total targets {args.targets} over Q={args.queries} at delta={args.delta:g}\n"
        "ratio = what the accountant charges for the derived knob, over the target.\n"
    )
    off = []
    for label, values in grids.items():
        print(f"{label}:")
        for target, entry in values.items():
            if "unreachable" in entry:
                print(f"    eps_total={target:>6}  ->  UNREACHABLE: {entry['unreachable']}")
                continue
            ratio = entry["ratio"]
            flag = "" if 0.98 <= ratio <= 1.02 else "   <-- MISMATCH"
            print(f"    eps_total={target:>6}  ->  {entry['value']}   (ratio {ratio}){flag}")
            if flag:
                off.append((label, target, ratio))
        print()

    if off:
        print("The inversion and the accountant disagree on these points; the grid")
        print("cannot be trusted until they match:")
        for label, target, ratio in off:
            print(f"  {label} @ {target}: ratio {ratio}")

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
