from yt_whisper.prompts import resolve_prompt, PROMPTS


def test_general_returns_none():
    assert resolve_prompt("general") is None


def test_grc_returns_prompt_string():
    result = resolve_prompt("grc")
    assert isinstance(result, str)
    assert "NIST" in result
    assert "FedRAMP" in result


def test_infosec_returns_prompt_string():
    result = resolve_prompt("infosec")
    assert isinstance(result, str)
    assert "CVE" in result
    assert "MITRE ATT&CK" in result


def test_unknown_key_treated_as_custom_string():
    custom = "my custom vocabulary terms"
    assert resolve_prompt(custom) == custom


def test_prompts_dict_has_expected_keys():
    assert set(PROMPTS.keys()) == {"general", "grc", "infosec"}
