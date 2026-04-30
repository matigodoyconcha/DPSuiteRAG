"""Smoke tests for bench_llms — verifies loading, config, and basic generation.

LLM_CANDIDATES: list of models to test. Add here when a new model is validated in the project.
Each model is downloaded on first run.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bench_llms import load_llm

# Add models here as they are validated in the project
LLM_CANDIDATES = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


@pytest.fixture(scope="module", params=LLM_CANDIDATES)
def llm(request):
    model_name = request.param
    return load_llm({"name": "huggingface", "model_name": model_name})


def test_llm_get_config_has_required_keys(llm):
    cfg = llm.get_config()
    assert "name" in cfg
    assert "model_name" in cfg


def test_llm_get_config_name_is_string(llm):
    cfg = llm.get_config()
    assert isinstance(cfg["name"], str)
    assert isinstance(cfg["model_name"], str)


def test_llm_build_generator_returns_generator(llm):
    from core.generator import BaseGenerator
    gen = llm.build_generator(max_new_tokens=8)
    assert isinstance(gen, BaseGenerator)


def test_llm_generate_returns_string(llm):
    gen = llm.build_generator(max_new_tokens=8)
    result = gen.generate("Hola, ¿cómo estás?")
    assert isinstance(result, str)


def test_llm_generate_nonempty_output(llm):
    gen = llm.build_generator(max_new_tokens=8)
    result = gen.generate("Completa esta frase: El cielo es")
    assert len(result) > 0


def test_load_llm_unsupported_raises():
    with pytest.raises(ValueError):
        load_llm({"name": "modelo_inexistente", "model_name": "x"})
