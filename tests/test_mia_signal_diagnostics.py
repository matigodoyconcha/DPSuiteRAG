"""An MIA cell that measured nothing must not read as an MIA cell that found nothing.

Both produce an AUC near 0.5. Only the diagnostics separate them, so each check here
pins one of the two readings and would fail if the fields stopped distinguishing them.
"""
from metrics.mia import (
    MIN_DISTINCT_SCORE_RATE,
    MIN_YES_NO_RATE,
    score_mia_records,
)


def _records(answers, anderson, gold="the capital of france is paris"):
    """One probe per answer, members first, alternating labels over the two halves."""
    half = len(answers) // 2
    return [
        {
            "is_member": i < half,
            "query": "q",
            "gold": gold,
            "doc": "d",
            "s2mia_answer": ans,
            "anderson_answer": and_,
        }
        for i, (ans, and_) in enumerate(zip(answers, anderson))
    ]


def test_noise_answers_are_flagged_non_informative():
    # What DP-RAG emitted at a derived per-token budget of 0.46: every reply is
    # unrelated tokens, so every BLEU collapses onto the same value.
    noise = ["Github tru nails ticking box yiplugin behalf"] * 40
    records = _records(noise, noise)
    m = score_mia_records(records)

    assert m["s2mia_distinct_score_rate"] < MIN_DISTINCT_SCORE_RATE
    assert m["s2mia_informative"] == 0.0
    # The AUC itself still lands at chance — which is exactly why it cannot be the
    # thing that carries the warning.
    assert abs(m["s2mia_auc"] - 0.5) < 0.25


def test_separating_answers_are_flagged_informative():
    members = [f"the capital of france is paris {i}" for i in range(20)]
    non_members = [f"unrelated text number {i} about weather" for i in range(20)]
    yes_no = ["Yes"] * 20 + ["No"] * 20
    m = score_mia_records(_records(members + non_members, yes_no))

    assert m["s2mia_distinct_score_rate"] >= MIN_DISTINCT_SCORE_RATE
    assert m["s2mia_informative"] == 1.0
    assert m["s2mia_auc"] > 0.9


def test_unparseable_anderson_replies_are_flagged_not_counted_as_no():
    # `yes_no_score` maps a reply with no yes/no token to 0.0, the same value an
    # explicit "No" gets, so silence would otherwise read as confident non-membership.
    garbage = ["painted constitutionacyBA things seulement"] * 40
    m = score_mia_records(_records(garbage, garbage))

    assert m["rag_mia_yes_no_rate"] == 0.0
    assert m["rag_mia_informative"] == 0.0


def test_answered_yes_no_probes_are_informative():
    replies = ["Yes, it appears."] * 20 + ["No, it does not."] * 20
    m = score_mia_records(_records(["x"] * 40, replies))

    assert m["rag_mia_yes_no_rate"] == 1.0
    assert m["rag_mia_informative"] == 1.0
    assert m["rag_mia_yes_no_rate"] >= MIN_YES_NO_RATE


def test_diagnostics_absent_for_empty_records():
    assert score_mia_records([]) == {}
