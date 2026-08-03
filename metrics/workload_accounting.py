"""Common privacy accountant for the whole query workload.

Every method in the benchmark declares a privacy number, but those numbers are not
comparable as they stand: some are per query, some cover the entire corpus, some are
offline; the delta ranges over 1e-3, 1e-4 and 0; and each method composes in whichever
family its own paper picked. This module puts all of them on one axis:

    epsilon_total(Q) := worst-case privacy loss with respect to one corpus document,
                        accumulated over a workload of Q queries, at a common delta.

The accounting is done as an **RDP curve plus an additive delta**, converted to
(epsilon, delta) exactly once at the end:

    epsilon_total(Q, delta) = min over alpha of [
          Q * sum_of_mechanism_RDP_curves(alpha)
        + BBGHS_conversion_term(alpha, delta_conversion) ]

    with  delta_conversion = delta - Q * delta_additive_per_query

Two properties make this the right common denominator:

* Every mechanism in the benchmark has a known RDP curve — Gaussian exactly
  (alpha * (sensitivity/sigma)^2 / 2), the exponential mechanism through the tight
  sinh bound, and any pure eps-DP mechanism through min(eps, alpha * eps^2 / 2).
  No method is forced to degrade into a family that does not fit it.
* RDP composes by adding curves, so the workload dimension Q is a multiplication
  rather than a re-derivation, and the inverse map (given a workload budget, what
  per-query budget does each method get) is a one-dimensional monotone root find.

Conversion is lossy and one-directional, so a mechanism is never round-tripped.
Two mechanisms sit outside the RDP path on purpose:

* Pure eps0-DP with delta = 0 (Private-RAG) satisfies (eps0, delta)-DP for every
  delta, so its number is reported directly. Pushing it through zCDP and back would
  turn eps0 = 10 into roughly 98 at delta = 1e-5 — an artefact of the idiom, not of
  the mechanism.
* Mechanisms whose guarantee is already an exact PLD composition (DP-RAG) can report
  that tighter number as a secondary column; the common-axis number stays RDP so the
  comparison between methods is like for like.

Approximate-DP mechanisms contribute a delta that no Renyi curve can absorb — the
bottom-out symbol of Limited-Domain Gumbel, and the false-release probability of
Propose-Test-Release. That part is tracked separately and subtracted from the target
before the conversion, which is what makes the per-token delta of the vote-based
methods visible: at T = 50 tokens and Q = 100 queries, a per-token delta of 1e-6
accumulates to 5e-3, five hundred times a 1e-5 target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence

from metrics.privacy_accounting_autodp import _rdp_em, _rdp_to_approx_dp

__all__ = [
    "Mechanism",
    "pure_dp",
    "exponential_mechanism",
    "gaussian",
    "zcdp",
    "WorkloadEpsilon",
    "epsilon_total",
    "epsilon_query_for_workload",
    "group_privacy",
    "dp_rag_mechanisms",
    "dpvoterag_mechanisms",
    "dpsparsevoterag_mechanisms",
    "dpksa_mechanisms",
    "private_rag_mechanisms",
    "dp_synrag_mechanisms",
    "METHOD_MECHANISMS",
    "METHOD_KNOB_BUILDERS",
    "METHOD_REGIMES",
    "budget_for_target",
    "method_epsilon_total",
    "DEFAULT_ALPHAS",
    "DEFAULT_DELTA",
]


DEFAULT_DELTA = 1e-2
"""Common delta for the whole workload.

Set by what the mechanisms can actually sustain, not by convention. Limited-Domain
Gumbel spends a fixed delta on every token it releases, so over Q=100 queries of 50
tokens DPVoteRAG spends 5e-3 of additive mass before any conversion slack; DP-KSA's
propose-test-release adds 5e-3 of its own. Nothing below about 1e-2 leaves room for
all seven, and the alternative — forcing a smaller number — does not buy privacy, it
raises those mechanisms' bottom-out thresholds until they stop answering (measured:
essentially every query halts on its first private token).

The honest reading of this value is that it is weak. For a 20 000-document corpus
1/n is 5e-5, so a delta of 1e-2 is two hundred times the point where the guarantee
stops excluding the release of whole documents. That is a property of composing
per-token approximate-DP mechanisms over a realistic workload, and it is the kind of
statement this axis exists to make: the source papers never compose their delta over
a query stream, so the cost has not been reported before. Methods whose delta is
merely conversion slack (DP-RAG, DP-SynRAG) can be read at a far tighter delta, and
Private-RAG is pure DP; those tighter figures belong alongside the common one, not
instead of it.
"""


def _default_alphas() -> tuple[float, ...]:
    """Renyi orders scanned for the tightest bound.

    Fractional orders below 2 matter when the composed curve is small (few queries,
    generous per-query budget); the integer tail matters in the opposite regime.
    """
    fractional = [1 + 0.25 * i for i in range(1, 8)]  # 1.25 .. 2.75
    integral = [float(a) for a in range(3, 257)]
    return tuple(fractional + integral)


DEFAULT_ALPHAS: tuple[float, ...] = _default_alphas()


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Mechanism:
    """One base mechanism, repeated ``count`` times within a single query.

    ``kind`` selects the RDP curve:

    ``"pure"``
        A generic ``parameter``-DP mechanism (Laplace, Sparse Vector, a per-document
        budget filter). Its RDP curve is ``min(eps, alpha * eps^2 / 2)`` — the plain
        pure-DP bound below the crossover, the Bun-Steinke zCDP bound above it.
    ``"exponential"``
        An exponential / Gumbel-max mechanism with pure-DP parameter ``parameter``.
        Uses the tight sinh curve, which is what makes composing many of them over a
        long token sequence affordable.
    ``"gaussian"``
        ``parameter`` is the noise multiplier ``sensitivity / sigma``; the curve
        ``alpha * parameter^2 / 2`` is exact.

    ``delta_add`` is the approximate-DP mass this mechanism contributes per instance
    and cannot be folded into the curve.
    """

    name: str
    kind: str
    parameter: float
    count: int = 1
    delta_add: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"pure", "exponential", "gaussian"}:
            raise ValueError(f"unknown mechanism kind {self.kind!r}")
        if self.parameter <= 0:
            raise ValueError(f"{self.name}: parameter must be > 0, got {self.parameter}")
        if self.count < 0:
            raise ValueError(f"{self.name}: count must be >= 0, got {self.count}")
        if not 0.0 <= self.delta_add < 1.0:
            raise ValueError(f"{self.name}: delta_add must be in [0, 1), got {self.delta_add}")

    def rdp(self, alpha: float) -> float:
        """RDP curve of ``count`` independent instances at Renyi order ``alpha``."""
        if self.count == 0:
            return 0.0
        if alpha <= 1.0:
            return math.inf
        eps = self.parameter
        if self.kind == "gaussian":
            single = alpha * eps * eps / 2.0
        elif self.kind == "exponential":
            single = _rdp_em(alpha, eps)
        else:
            single = min(eps, 0.5 * alpha * eps * eps)
        return self.count * single

    @property
    def total_delta_add(self) -> float:
        return self.count * self.delta_add

    @property
    def pure_epsilon(self) -> float:
        """Pure-DP parameter of ``count`` instances, or infinity if there is none."""
        if self.count == 0:
            return 0.0
        if self.kind == "gaussian":
            return math.inf
        return self.count * self.parameter


def zcdp(name: str, rho: float, count: int = 1, delta_add: float = 0.0) -> Mechanism:
    """A rho-zCDP mechanism, expressed exactly as the RDP curve ``rho * alpha``.

    Lets an accountant written in zCDP (DP-SynRAG composes its four build stages that
    way) join the shared axis without any loss: zCDP is RDP with a linear curve, and
    ``rho * alpha`` is reproduced by a Gaussian of noise multiplier ``sqrt(2 * rho)``.
    Converting through this path is tighter than the ``rho + 2*sqrt(rho*ln(1/delta))``
    closed form, because the conversion is optimised over the Renyi order.
    """
    if rho <= 0:
        raise ValueError(f"{name}: rho must be > 0, got {rho}")
    return Mechanism(
        name=name,
        kind="gaussian",
        parameter=math.sqrt(2.0 * rho),
        count=count,
        delta_add=delta_add,
    )


def pure_dp(name: str, epsilon: float, count: int = 1, delta_add: float = 0.0) -> Mechanism:
    """A generic ``epsilon``-DP mechanism (Laplace, Sparse Vector, budget filter)."""
    return Mechanism(name=name, kind="pure", parameter=epsilon, count=count, delta_add=delta_add)


def exponential_mechanism(
    name: str, epsilon: float, count: int = 1, delta_add: float = 0.0
) -> Mechanism:
    """An exponential / Gumbel-max mechanism with pure-DP parameter ``epsilon``."""
    return Mechanism(
        name=name, kind="exponential", parameter=epsilon, count=count, delta_add=delta_add
    )


def gaussian(
    name: str, sigma: float, sensitivity: float = 1.0, count: int = 1, delta_add: float = 0.0
) -> Mechanism:
    """A Gaussian mechanism with noise scale ``sigma`` and L2 ``sensitivity``."""
    if sigma <= 0:
        raise ValueError(f"{name}: sigma must be > 0, got {sigma}")
    return Mechanism(
        name=name,
        kind="gaussian",
        parameter=sensitivity / sigma,
        count=count,
        delta_add=delta_add,
    )


# ---------------------------------------------------------------------------
# Workload composition
# ---------------------------------------------------------------------------

class WorkloadEpsilon(NamedTuple):
    """Privacy of a Q-query workload, on the common axis."""

    epsilon: float
    """Headline epsilon at ``delta``, via RDP composition."""

    delta: float
    """Target delta covering the whole workload."""

    queries: int
    """Workload size Q the number refers to."""

    best_alpha: float
    """Renyi order that minimised the bound."""

    delta_additive: float
    """Approximate-DP mass spent by the mechanisms themselves (Q * per-query)."""

    delta_conversion: float
    """Delta left for the RDP -> DP conversion."""

    epsilon_basic: float
    """Control column: the same workload under basic composition (Q * per-query eps)."""

    regime: str
    """``"per_query"``, ``"corpus_total"`` or ``"offline"``."""

    components: dict
    """Per-mechanism RDP contribution at ``best_alpha``, for run logs."""


def _compose_rdp(mechanisms: Sequence[Mechanism], alpha: float, queries: int) -> float:
    total = 0.0
    for mech in mechanisms:
        total += mech.rdp(alpha)
        if math.isinf(total):
            return math.inf
    return queries * total


def epsilon_total(
    mechanisms: Sequence[Mechanism],
    *,
    queries: int,
    delta: float = DEFAULT_DELTA,
    regime: str = "per_query",
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> WorkloadEpsilon:
    """Compose one query's mechanisms over a ``queries``-query workload.

    ``regime`` documents how the cost scales. ``"per_query"`` composes the mechanism
    list ``queries`` times. ``"corpus_total"`` and ``"offline"`` describe methods whose
    guarantee already covers the whole stream (a per-document budget filter, an offline
    synthetic corpus build); for those the list is composed once and ``queries`` only
    labels the result.
    """
    if queries < 1:
        raise ValueError(f"queries must be >= 1, got {queries}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if regime not in {"per_query", "corpus_total", "offline"}:
        raise ValueError(f"unknown regime {regime!r}")
    if not mechanisms or all(m.count == 0 for m in mechanisms):
        # An empty inventory would compose to epsilon = 0, i.e. a claim of perfect
        # privacy for a method whose mechanisms simply were not described.
        raise ValueError(
            "no mechanisms to compose: a method's inventory must list at least one "
            "mechanism with a positive count."
        )

    repeats = queries if regime == "per_query" else 1
    delta_additive = repeats * sum(m.total_delta_add for m in mechanisms)
    delta_conversion = delta - delta_additive
    if delta_conversion <= 0.0:
        raise ValueError(
            f"the mechanisms' own delta ({delta_additive:.3g} over {repeats} "
            f"repetition(s)) already exhausts the target delta ({delta:.3g}). "
            "Lower the per-instance delta_add — for a T-token, Q-query workload it "
            "must be below delta / (T * Q)."
        )

    best_eps = math.inf
    best_alpha = math.nan
    for alpha in alphas:
        rdp = _compose_rdp(mechanisms, alpha, repeats)
        # The sinh form of the exponential-mechanism curve cancels catastrophically at
        # a very small epsilon and can come out zero or slightly negative. That is
        # numerical noise, not a privacy claim: converting it would report epsilon = 0
        # exactly, i.e. perfect privacy for a mechanism that has none. Such an order
        # carries no information, so it is skipped and the sequential bound below
        # supplies the answer.
        if not math.isfinite(rdp) or rdp <= 0.0:
            continue
        eps = _rdp_to_approx_dp(rdp, alpha, delta_conversion)
        if eps < best_eps:
            best_eps = eps
            best_alpha = alpha

    # Sequential composition of the pure-DP parameters is a valid bound in its own
    # right, and it wins whenever the conversion slack costs more than it saves —
    # a short workload, or a single mechanism whose guarantee is already pure. Taking
    # the minimum is what keeps a pure eps0-DP method from being inflated by an idiom
    # it never needed: eps0 = 10 stays 10 instead of becoming ~98 through a zCDP
    # round trip. Both paths are sound, so the tighter one is reported.
    pure_total = sum(m.pure_epsilon for m in mechanisms)
    if math.isfinite(pure_total) and repeats * pure_total < best_eps:
        best_eps = repeats * pure_total
        best_alpha = math.inf

    if not math.isfinite(best_eps):
        raise RuntimeError(
            "no Renyi order produced a finite epsilon; widen `alphas` or check the "
            "mechanism parameters."
        )

    # Basic composition control: convert one query at delta/repeats, then multiply.
    # Reported alongside the headline so a ranking can be checked against the
    # loosest defensible accounting rather than taken on trust.
    single = _single_query_epsilon(mechanisms, delta / repeats, alphas)
    epsilon_basic = repeats * single

    components = {
        m.name: (m.rdp(best_alpha) if math.isfinite(best_alpha) else m.pure_epsilon)
        for m in mechanisms
    }
    return WorkloadEpsilon(
        epsilon=best_eps,
        delta=delta,
        queries=queries,
        best_alpha=best_alpha,
        delta_additive=delta_additive,
        delta_conversion=delta_conversion,
        epsilon_basic=epsilon_basic,
        regime=regime,
        components=components,
    )


def _single_query_epsilon(
    mechanisms: Sequence[Mechanism], delta: float, alphas: Sequence[float]
) -> float:
    """(epsilon, delta) of one query, used by the basic-composition control column."""
    delta_add = sum(m.total_delta_add for m in mechanisms)
    delta_conv = delta - delta_add
    if delta_conv <= 0.0:
        return math.inf
    best = math.inf
    for alpha in alphas:
        rdp = _compose_rdp(mechanisms, alpha, 1)
        if math.isinf(rdp):
            continue
        best = min(best, _rdp_to_approx_dp(rdp, alpha, delta_conv))
    pure_total = sum(m.pure_epsilon for m in mechanisms)
    return min(best, pure_total)


# ---------------------------------------------------------------------------
# Inverse map: workload budget -> per-query budget
# ---------------------------------------------------------------------------

def epsilon_query_for_workload(
    build: Callable[[float], Sequence[Mechanism]],
    *,
    target_epsilon: float,
    queries: int,
    delta: float = DEFAULT_DELTA,
    regime: str = "per_query",
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    lower: float = 1e-6,
    upper: float = 1e4,
    iterations: int = 200,
) -> float:
    """Largest per-query budget whose Q-query workload stays within ``target_epsilon``.

    ``build`` maps a candidate per-query epsilon to that query's mechanism list, which
    is what lets each method keep its own internal split (retrieval versus generation,
    threshold versus vote) while sitting on the shared axis.

    The composed epsilon is monotone in the per-query budget, so this is a bisection on
    a log scale. It always returns from the feasible side: the workload of the returned
    value is at or below the target, never above.
    """
    if target_epsilon <= 0:
        raise ValueError(f"target_epsilon must be > 0, got {target_epsilon}")

    def achieved(eps_query: float) -> float:
        try:
            return epsilon_total(
                build(eps_query),
                queries=queries,
                delta=delta,
                regime=regime,
                alphas=alphas,
            ).epsilon
        except (ValueError, RuntimeError):
            return math.inf

    if achieved(lower) > target_epsilon:
        raise ValueError(
            f"even a per-query budget of {lower:.3g} exceeds the workload target "
            f"{target_epsilon:.3g} at Q={queries}. The mechanism's own additive delta "
            "or its token count is the binding constraint, not epsilon."
        )
    if achieved(upper) <= target_epsilon:
        return upper

    lo, hi = lower, upper
    for _ in range(iterations):
        mid = math.sqrt(lo * hi)
        if achieved(mid) <= target_epsilon:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Per-method mechanism inventories
# ---------------------------------------------------------------------------
#
# Each builder takes a single scalar budget knob plus the method's structural
# parameters, and returns the mechanisms one query runs. Scaling a single knob —
# rather than searching the method's whole hyperparameter grid — keeps the internal
# split each paper chose (retrieval versus generation, threshold versus vote) fixed
# a priori, so the workload axis measures the mechanism and not a per-point tuning.

def dp_rag_mechanisms(
    *, epsilon_retrieval: float, token_epsilon: float, max_tokens: int
) -> list[Mechanism]:
    """DP-RAG: one retrieval exponential mechanism plus T per-token ones.

    ``max_tokens`` is the configured decoding ceiling, not the length a run produced.
    """
    return [
        exponential_mechanism("retrieval", epsilon_retrieval),
        exponential_mechanism("generation", token_epsilon, count=max_tokens),
    ]


def dpvoterag_mechanisms(
    *, epsilon_token: float, max_tokens: int, delta_token: float, n_peels: int = 1
) -> list[Mechanism]:
    """DPVoteRAG: T Limited-Domain Gumbel votes, each with a bottom-out delta.

    ``max_tokens`` is the configured decoding ceiling, not the length a run produced:
    an answer that stops early spends less, but the guarantee has to hold for the
    longest answer the mechanism could have emitted on a neighbouring corpus.
    """
    return [
        exponential_mechanism(
            "vote", epsilon_token, count=n_peels * max_tokens, delta_add=delta_token
        )
    ]


def dpsparsevoterag_mechanisms(
    *,
    max_private_tokens: int,
    epsilon_token: float,
    delta_token: float,
    sparse_epsilon: float,
    n_epochs: int = 1,
    n_peels: int = 1,
) -> list[Mechanism]:
    """DPSparseVoteRAG: the vote sequence plus one Sparse Vector epoch per threshold.

    ``max_private_tokens`` is the pre-specified ceiling ``c`` on private releases
    (Algorithm 3), never the number a run turned out to make. How many tokens the
    sparse test actually releases depends on the data — a corpus may yield three
    releases and its neighbour twenty — so charging the realised count produces a
    data-dependent number that is not a uniform bound over neighbouring corpora. It
    would read as a far smaller epsilon precisely on the runs that halt early.
    """
    return [
        pure_dp("sparse_vector", sparse_epsilon, count=n_epochs),
        exponential_mechanism(
            "vote",
            epsilon_token,
            count=n_peels * max_private_tokens,
            delta_add=delta_token,
        ),
    ]


def dpksa_mechanisms(
    *,
    gumbel_scale: float,
    sigma: float,
    delta_ptr: float,
    gap_sensitivity: float = 2.0,
) -> list[Mechanism]:
    """DP-KSA: the FindBestK exponential mechanism plus the Gaussian PTR test.

    ``delta_ptr`` is the false-release probability, which is approximate-DP mass no
    Renyi curve absorbs — at a hundred queries it is charged a hundred times.
    """
    return [
        exponential_mechanism("find_best_k", 2.0 * gap_sensitivity / gumbel_scale),
        gaussian("ptr_test", sigma=sigma, sensitivity=1.0, delta_add=delta_ptr),
    ]


def private_rag_mechanisms(*, m_retrievals: int, epsilon_q: float) -> list[Mechanism]:
    """Private-RAG: the per-document budget filter, pure DP and flat in the workload.

    A document serves at most ``M`` queries before it switches off, so the corpus-level
    cost is ``M * eps_q`` no matter how long the stream runs. Use it with
    ``regime="corpus_total"``.
    """
    return [pure_dp("document_filter", m_retrievals * epsilon_q)]


def dp_synrag_mechanisms(*, rho: float) -> list[Mechanism]:
    """DP-SynRAG: the whole offline corpus build, as its composed zCDP parameter.

    Answering queries against the synthetic corpus is post-processing, so the cost is
    flat in the workload. Use it with ``regime="offline"``.
    """
    return [zcdp("corpus_build", rho)]


METHOD_REGIMES: dict[str, str] = {
    "dp_rag": "per_query",
    "dpvoterag": "per_query",
    "dpsparsevoterag": "per_query",
    "dpksa": "per_query",
    "private_rag": "corpus_total",
    "private_rag_ada": "corpus_total",
    "dp_synrag": "offline",
}
"""How each method's privacy cost scales with the number of queries."""


METHOD_MECHANISMS: dict[str, Callable[..., list[Mechanism]]] = {
    "dp_rag": dp_rag_mechanisms,
    "dpvoterag": dpvoterag_mechanisms,
    "dpsparsevoterag": dpsparsevoterag_mechanisms,
    "dpksa": dpksa_mechanisms,
    "private_rag": private_rag_mechanisms,
    "private_rag_ada": private_rag_mechanisms,
    "dp_synrag": dp_synrag_mechanisms,
}


def method_epsilon_total(
    method: str,
    *,
    queries: int,
    delta: float = DEFAULT_DELTA,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    **params,
) -> WorkloadEpsilon:
    """Workload epsilon for a named method, on the common axis."""
    try:
        build = METHOD_MECHANISMS[method]
    except KeyError:
        raise ValueError(
            f"unknown method {method!r}; known methods: {sorted(METHOD_MECHANISMS)}"
        ) from None
    return epsilon_total(
        build(**params),
        queries=queries,
        delta=delta,
        regime=METHOD_REGIMES[method],
        alphas=alphas,
    )


# ---------------------------------------------------------------------------
# Group privacy
# ---------------------------------------------------------------------------

def group_privacy(epsilon: float, delta: float, group_size: int) -> tuple[float, float]:
    """Lift an (epsilon, delta) guarantee from one unit to a group of ``group_size``.

    Splitting a document into ``c`` chunks and running a mechanism that protects a
    single chunk yields a document-level guarantee of

        (c * eps,  c * exp((c - 1) * eps) * delta)

    The delta grows exponentially in ``c``, which is why the benchmark indexes whole
    documents: at c = 5 and eps = 1 a delta of 1e-5 already degrades to 2.7e-3, and no
    document-level statement worth reporting survives it.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    if group_size == 1:
        return float(epsilon), float(delta)
    eps_group = group_size * epsilon
    delta_group = group_size * math.exp((group_size - 1) * epsilon) * delta
    return eps_group, min(delta_group, 1.0)


def _dp_rag_knob(knob: float, *, retrieval_ratio: float, max_tokens: int):
    return dp_rag_mechanisms(
        epsilon_retrieval=retrieval_ratio * knob, token_epsilon=knob, max_tokens=max_tokens
    )


def _dpvoterag_knob(knob: float, *, max_tokens: int, delta_token: float):
    # The swept budget is the query's whole allowance, split across the answer.
    return dpvoterag_mechanisms(
        epsilon_token=knob / max_tokens, max_tokens=max_tokens, delta_token=delta_token
    )


def _dpsparsevoterag_knob(
    knob: float, *, max_private_tokens: int, delta_token: float, sparse_fraction: float
):
    per_token = knob / max_private_tokens
    return dpsparsevoterag_mechanisms(
        epsilon_token=(1.0 - sparse_fraction) * per_token,
        max_private_tokens=max_private_tokens,
        delta_token=delta_token,
        sparse_epsilon=sparse_fraction * per_token,
        n_epochs=1 + max_private_tokens,
    )


def _dpksa_knob(knob: float, *, mechanism_delta: float, ptr_noise_coupling: float):
    # Named apart from the shared axis's delta on purpose: this one calibrates sigma
    # and the propose-test-release threshold, the other is the delta the workload is
    # reported at. They are different quantities, and a caller that has to pass both
    # cannot if they share a keyword.
    from metrics.privacy_accounting_dpksa import calibrate_dpksa

    cal = calibrate_dpksa(
        target_epsilon=knob, target_delta=mechanism_delta,
        ptr_noise_coupling=ptr_noise_coupling,
    )
    return dpksa_mechanisms(
        gumbel_scale=cal["gumbel_scale"], sigma=cal["sigma"], delta_ptr=cal["delta_ptr"]
    )


def _private_rag_knob(knob: float, *, m_retrievals: int):
    return private_rag_mechanisms(m_retrievals=m_retrievals, epsilon_q=knob)


def _dp_synrag_knob(knob: float, **_):
    # The four stages scale together; the knob is the resulting zCDP rho.
    return dp_synrag_mechanisms(rho=knob)


#: How each method's own budget knob turns into the mechanisms it charges. Shared by
#: the runners, which derive their grid, and by analysis/derive_epsilon_grids.py, which
#: reports it — one definition, so a sweep and its published grid cannot disagree.
METHOD_KNOB_BUILDERS: dict[str, Callable[..., list[Mechanism]]] = {
    "dp_rag": _dp_rag_knob,
    "dpvoterag": _dpvoterag_knob,
    "dpsparsevoterag": _dpsparsevoterag_knob,
    "dpksa": _dpksa_knob,
    "private_rag": _private_rag_knob,
    "private_rag_ada": _private_rag_knob,
    "dp_synrag": _dp_synrag_knob,
}


def _dpksa_budget_floor(*, mechanism_delta: float, ptr_noise_coupling: float) -> float:
    """Smallest budget DP-KSA's calibration can actually realise.

    Below it `calibrate_dpksa` cannot hit the target and raises, which a bisection
    starting from 1e-6 reads as "this cell is unreachable" — reporting the calibrator's
    own floor as a property of the accounting and marking every cell out of reach.
    """
    from metrics.privacy_accounting_dpksa import calibrate_dpksa

    lo, hi = 1e-6, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        try:
            calibrate_dpksa(
                target_epsilon=mid, target_delta=mechanism_delta,
                ptr_noise_coupling=ptr_noise_coupling,
            )
        except Exception:
            lo = mid
        else:
            hi = mid
    return hi


def budget_for_target(
    method: str,
    *,
    target_epsilon: float,
    queries: int,
    delta: float = DEFAULT_DELTA,
    **structural,
) -> float:
    """The method's own budget knob that reaches ``target_epsilon`` over ``queries``.

    This is what makes the workload size an axis you can sweep. A per-query method
    serving a longer stream has to spend less on each question to land on the same
    total, so the knob moves with Q; a method whose guarantee already covers the
    stream does not move at all. Running both at a fixed target is what turns that
    difference into a measured utility curve rather than an argument about accounting.

    Raises when the target is out of reach — a method whose additive delta or whose
    token count already exceeds it has no budget that would do, and reporting a
    clamped value would put a point on the axis that the mechanism never honoured.
    """
    if method not in METHOD_KNOB_BUILDERS:
        raise KeyError(f"no knob builder for {method!r}; known: {sorted(METHOD_KNOB_BUILDERS)}")
    build = METHOD_KNOB_BUILDERS[method]
    # DP-KSA's calibration does not exist below a floor of its own, so the search has
    # to start above it; every other method is well defined all the way down.
    lower = (
        _dpksa_budget_floor(
            mechanism_delta=structural["mechanism_delta"],
            ptr_noise_coupling=structural["ptr_noise_coupling"],
        )
        if method == "dpksa"
        else 1e-6
    )
    return epsilon_query_for_workload(
        lambda knob: build(knob, **structural),
        target_epsilon=target_epsilon,
        queries=queries,
        delta=delta,
        regime=METHOD_REGIMES.get(method, "per_query"),
        lower=lower,
    )
