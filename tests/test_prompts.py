"""Tests for prompt profile resolution and keyword validation."""

import pytest
from yt_whisper.prompts import PROMPTS, resolve_prompt


def test_resolve_known_profile_returns_text():
    result = resolve_prompt("grc")
    assert isinstance(result, str)
    assert "NIST" in result


def test_resolve_general_returns_none_or_empty():
    # general has no prompt text -- must be None for whisper
    result = resolve_prompt("general")
    assert result is None


def test_resolve_infosec_returns_text():
    result = resolve_prompt("infosec")
    assert isinstance(result, str)
    assert "CVE" in result


def test_resolve_custom_string_passthrough():
    custom = "my custom prompt about widgets"
    assert resolve_prompt(custom) == custom


def test_all_profiles_have_keywords_list():
    for name, profile in PROMPTS.items():
        assert "keywords" in profile, f"{name} missing keywords"
        assert isinstance(profile["keywords"], list)


def test_general_has_empty_keywords():
    assert PROMPTS["general"]["keywords"] == []


def test_grc_has_keywords():
    assert len(PROMPTS["grc"]["keywords"]) > 0
    assert "NIST" in PROMPTS["grc"]["keywords"]


def test_infosec_has_keywords():
    assert len(PROMPTS["infosec"]["keywords"]) > 0
    assert "CVE" in PROMPTS["infosec"]["keywords"]


def test_no_duplicate_keywords_within_profile():
    for name, profile in PROMPTS.items():
        kws = profile["keywords"]
        assert len(kws) == len(set(kws)), f"{name} has duplicate keywords"


def test_no_empty_string_keywords():
    for name, profile in PROMPTS.items():
        for kw in profile["keywords"]:
            assert kw.strip() != "", f"{name} has empty keyword"
