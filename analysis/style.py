"""One visual identity for every figure the analysis module draws.

Each method keeps the same name, colour and marker in every figure, whichever script
draws it. The method colours are the Okabe-Ito palette, which stays distinguishable
under the common forms of colour blindness, and every method also has its own marker
so the figures still read in greyscale print.

    from analysis.style import METHOD_STYLE, method_label, apply
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import matplotlib

#: DP method key -> (display name, colour, marker), in legend order. The keys are the
#: method names the runners write (`analysis.load_results.method_of_payload`).
METHOD_STYLE: dict[str, tuple[str, str, str]] = {
    "dp_rag": ("DP-RAG", "#0072B2", "o"),
    "dpvoterag": ("DPVoteRAG", "#009E73", "s"),
    "dpsparsevoterag": ("DPSparseVoteRAG", "#D55E00", "^"),
    "dpksa": ("DP-KSA", "#CC79A7", "D"),
    "private_rag": ("MuRAG", "#56B4E9", "v"),
    "private_rag_ada": ("MuRAG-Ada", "#7B3294", "P"),
    "dp_synrag": ("DP-SynRAG", "#E69F00", "X"),
}

#: Methods drawn wider, underneath the rest: (line width, marker size). On many cells
#: MuRAG and MuRAG-Ada give the same value, and MuRAG would otherwise be hidden.
UNDERLAY: dict[str, tuple[float, float]] = {"private_rag": (4.0, 8.5)}


def line_kwargs(method: str) -> dict:
    """Line and marker of a method, as keyword arguments for `Axes.plot`."""
    label, colour, marker = METHOD_STYLE[method]
    lw, ms = UNDERLAY.get(method, (1.7, 5.0))
    return {"color": colour, "marker": marker, "lw": lw, "ms": ms,
            "mec": "white", "mew": 0.7, "zorder": 2.5 if method in UNDERLAY else 3}


#: The two epsilon-free conditions every DP method is read against.
FLOOR_METHOD = "no_rag"
CEILING_METHOD = "vanilla"
REFERENCE_STYLE: dict[str, tuple[str, str, str]] = {
    FLOOR_METHOD: ("No retrieval (floor)", "#8c8c8c", "s"),
    CEILING_METHOD: ("Non-private RAG (ceiling)", "#1a1a1a", "o"),
}

#: Corpus key -> display name. A corpus missing here is shown under its key.
DATASET_LABELS: dict[str, str] = {
    "medical_dirichlet_phi3": "Medical (synthetic)",
    "triviaqa": "TriviaQA",
    "natural_questions": "Natural Questions",
    "searchqa": "SearchQA",
    "chatdoctor": "ChatDoctor",
    "movielens": "MovieLens",
    "squad": "SQuAD",
    "hotpotqa": "HotpotQA",
}

FLOOR_LABEL = REFERENCE_STYLE[FLOOR_METHOD][0]
CEILING_LABEL = REFERENCE_STYLE[CEILING_METHOD][0]
EPSILON_AXIS = "Privacy budget ε"
WORKLOAD_AXIS = "Workload size Q"

INK = "#1a1a1a"
MUTED = "#6b6b6b"
REF = "#8c8c8c"
GRID = "#ececec"
SPINE = "#c9c9c9"

RC = {
    "font.size": 11.0,
    "axes.titlesize": 12.0,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 10.0,
    "axes.titlecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "lines.linewidth": 1.7,
    "lines.markersize": 5.0,
    "savefig.dpi": 200,
}


def apply() -> None:
    """Set the shared typography (call once, before drawing)."""
    matplotlib.rcParams.update(RC)


def method_label(method: str) -> str:
    """Display name of a method or reference condition (its key when unknown)."""
    style = METHOD_STYLE.get(method) or REFERENCE_STYLE.get(method)
    return style[0] if style else method


def method_color(method: str, default: str = "#7f7f7f") -> str:
    style = METHOD_STYLE.get(method) or REFERENCE_STYLE.get(method)
    return style[1] if style else default


def method_marker(method: str, default: str = "o") -> str:
    style = METHOD_STYLE.get(method) or REFERENCE_STYLE.get(method)
    return style[2] if style else default


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def style_axes(ax) -> None:
    """Recessive frame: no top/right spines, pale grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, lw=0.9)
    ax.set_axisbelow(True)


def half_up(value: float, digits: int = 2) -> str:
    """Round half-up on the exact fraction the float approximates.

    Accuracies are means of k/40 over generators, so ties such as 0.0125 are common,
    and `format` would round them to the even digit.
    """
    exact = Fraction(value).limit_denominator(100000)
    d = Decimal(exact.numerator) / Decimal(exact.denominator)
    return str(d.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))
