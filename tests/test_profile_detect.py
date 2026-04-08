"""Tests for keyword-based prompt profile auto-detection."""

from yt_whisper.profile_detect import detect_profile, CONFIDENCE_THRESHOLD


def _meta(title="", channel="", description="", tags=None):
    return {
        "title": title,
        "channel": channel,
        "description": description,
        "tags": tags or [],
    }


def test_detect_grc_from_title():
    m = _meta(title="NIST CSF and SOC 2 Overview", description="compliance audit walkthrough")
    name, matched, conf = detect_profile(m)
    assert name == "grc"
    assert "NIST" in matched
    assert conf > 0


def test_detect_infosec_from_description():
    m = _meta(title="Deep Dive", description="We analyze CVE-2024-1234 and explore exploit techniques used by red team operators")
    name, matched, conf = detect_profile(m)
    assert name == "infosec"
    assert "CVE" in matched
    assert "exploit" in matched


def test_detect_below_threshold_returns_general():
    m = _meta(title="Cooking Pasta", description="We mention audit once in passing")
    name, matched, conf = detect_profile(m)
    assert name == "general"
    assert matched == []
    assert conf == 0.0


def test_detect_empty_metadata():
    name, matched, conf = detect_profile(_meta())
    assert name == "general"
    assert matched == []


def test_detect_missing_fields_does_not_crash():
    # All fields missing entirely
    name, matched, conf = detect_profile({})
    assert name == "general"


def test_detect_case_insensitive():
    m = _meta(description="cve-2024-1234 and red team analysis and exploit detail")
    name, _, _ = detect_profile(m)
    assert name == "infosec"


def test_detect_word_boundaries_prevent_false_match():
    # "cover" should not match "cve"
    m = _meta(description="We cover the covered topic in this coverage video about cover letters")
    name, matched, _ = detect_profile(m)
    assert "CVE" not in matched
    assert name == "general"


def test_detect_multi_word_keywords_weigh_more():
    # "red team" and "penetration testing" should outweigh single generic words
    m = _meta(description="red team penetration testing walkthrough")
    name, matched, _ = detect_profile(m)
    assert name == "infosec"
    assert "red team" in matched
    assert "penetration testing" in matched


def test_detect_description_cap_2000_chars():
    # A keyword past the 2000-char cap should not match
    padding = "x " * 1500  # > 2000 chars
    m = _meta(description=padding + " CVE")
    name, matched, _ = detect_profile(m)
    assert "CVE" not in matched


def test_detect_matched_terms_reported_for_display():
    m = _meta(title="NIST 800-53 and SOC 2 compliance")
    name, matched, _ = detect_profile(m)
    assert name == "grc"
    assert len(matched) >= 2


def test_detect_tags_contribute_to_matching():
    m = _meta(title="Talk", tags=["CVE", "exploit", "red team"])
    name, _, _ = detect_profile(m)
    assert name == "infosec"


def test_detect_confidence_between_zero_and_one():
    m = _meta(title="NIST SOC 2 compliance audit framework control", description="FedRAMP HIPAA GDPR")
    _, _, conf = detect_profile(m)
    assert 0.0 <= conf <= 1.0


def test_confidence_threshold_is_defined():
    assert isinstance(CONFIDENCE_THRESHOLD, int)
    assert CONFIDENCE_THRESHOLD >= 1
