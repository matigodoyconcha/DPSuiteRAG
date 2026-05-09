"""DP configuration presets for DPVoteRAG / DPSparseVoteRAG.

Two built-in presets:
  'paper' — mirrors tacchan7412/DPRAG settings for numerical comparability.
  'local' — preserves the pre-refactor defaults (backward compatible).

Usage in runners:
    from dp_methods.dpvoterag.presets import resolve_dp_config
    dp_cfg = resolve_dp_config(config.get("dp", {}))
    generator = VoteRAGGenerator(..., **dp_cfg)
"""

from __future__ import annotations

PRESETS: dict[str, dict] = {
    "paper": {
        # Issue 1: tight EM-peeling accountant (autodp)
        "accounting_method": "em_peeling_autodp",
        # Issue 12: no subsampling in paper experiments
        "subsampling_rate": 1.0,
        # Issue 2: paper uses scale=1/eps (no sensitivity scaling in Gumbel noise)
        "sensitivity": 2.0,
        "gumbel_scale_uses_sens": False,
        # Issue 8: full LD domain (including unobserved vocab tokens)
        "ld_extend_to_vocab": True,
        # Issue 6: paper uses fail_mode='stop'
        "fail_mode": "stop",
        # Issue 5: token-ID append (no re-tokenization)
        "prompt_construction": "token_ids_append",
        # Issue 11: no <bos> prepended on each call
        "add_special_tokens_initial": False,
        # Issue 7: paper prompt template
        "prompt_template": "paper",
        # Issue 4: accountant drives stopping; tmax_absolute is only a safety net
        "tmax_strategy": "accountant_only",
        "tmax_absolute": 50,
        # Issue 3 (SparseVote only)
        "sparse_use_online_budget": True,
        "sparse_account_svt": True,
        # Issue 10 (SparseVote only)
        "stopwords_format": "paper_pt",
        "stopwords_path": "data/stopword_words.pt",
        "stopwords_token_index": 1,
    },
    "local": {
        # Prior defaults — reproduce behavior before this refactor
        "accounting_method": "simple_bound",
        "subsampling_rate": 1.0,
        "sensitivity": 2.0,
        "gumbel_scale_uses_sens": True,
        "ld_extend_to_vocab": False,
        "fail_mode": "ld_pate",
        "prompt_construction": "string_retokenize",
        "add_special_tokens_initial": True,
        "prompt_template": "local",
        "tmax_strategy": "legacy_max",
        "tmax_absolute": 200,
        "sparse_use_online_budget": False,
        "sparse_account_svt": False,
        "stopwords_format": "local_txt",
        "stopwords_path": None,
        "stopwords_token_index": None,
    },
}

# Keys that only apply to SparseVoteRAGGenerator; stripped when building VoteRAGGenerator
_SPARSE_ONLY_KEYS = {"sparse_use_online_budget", "sparse_account_svt", "stopwords_format",
                     "stopwords_path", "stopwords_token_index"}


def resolve_dp_config(dp_yaml_section: dict) -> dict:
    """Merge preset defaults with any YAML overrides.

    If dp_yaml_section contains 'preset': 'paper' or 'local', start from that
    preset and apply any additional keys as overrides.  If preset is 'custom'
    or absent, return the section as-is.

    Args:
        dp_yaml_section: The 'dp:' block from the experiment YAML.

    Returns:
        Flat dict of DP configuration kwargs ready to unpack into a generator
        constructor.
    """
    preset_name = dp_yaml_section.get("preset", "custom")
    if preset_name == "custom" or preset_name not in PRESETS:
        result = dict(dp_yaml_section)
        result.pop("preset", None)
        return result

    base = PRESETS[preset_name].copy()
    overrides = {k: v for k, v in dp_yaml_section.items() if k != "preset"}
    base.update(overrides)
    return base


def resolve_dp_config_for_voterag(dp_yaml_section: dict) -> dict:
    """Like resolve_dp_config but strips sparse-only keys for VoteRAGGenerator."""
    cfg = resolve_dp_config(dp_yaml_section)
    return {k: v for k, v in cfg.items() if k not in _SPARSE_ONLY_KEYS}
