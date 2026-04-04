"""Named domain vocabulary prompt profiles for Whisper transcription."""

PROMPTS = {
    "general": None,
    "grc": (
        "NIST, RMF, Risk Management Framework, CMMC, FedRAMP, SOC2, SOC 2, GRC, "
        "cybersecurity, compliance, audit, control, framework, assessment, authorization, "
        "ATO, FISMA, FIPS 199, SP 800-53, SP 800-37, risk register, risk assessment, "
        "threat modeling, likelihood, impact, inherent risk, residual risk, Gerald Auger"
    ),
    "infosec": (
        "CVE, CVSS, vulnerability, exploit, zero-day, malware, ransomware, phishing, "
        "SOC, SIEM, EDR, XDR, MITRE ATT&CK, threat intelligence, incident response, "
        "penetration testing, red team, blue team, OSINT, IOC, indicators of compromise"
    ),
}


def resolve_prompt(name_or_string):
    """Return prompt text. Known key -> stored value. Unknown key -> treat as custom string."""
    return PROMPTS.get(name_or_string, name_or_string)
