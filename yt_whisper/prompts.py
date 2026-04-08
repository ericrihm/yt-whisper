"""Named domain vocabulary prompt profiles for Whisper transcription.

Each profile has:
- text: the initial_prompt passed to whisper (or None for no hint)
- keywords: list of terms used by profile_detect.py to auto-select this profile
"""

PROMPTS = {
    "general": {
        "text": None,
        "keywords": [],
    },
    "grc": {
        "text": (
            "NIST, RMF, Risk Management Framework, CMMC, FedRAMP, SOC2, SOC 2, GRC, "
            "cybersecurity, compliance, audit, control, framework, assessment, authorization, "
            "ATO, FISMA, FIPS 199, SP 800-53, SP 800-37, risk register, risk assessment, "
            "threat modeling, likelihood, impact, inherent risk, residual risk, Gerald Auger"
        ),
        "keywords": [
            "NIST", "RMF", "CMMC", "FedRAMP", "SOC 2", "SOC2", "ISO 27001",
            "HIPAA", "PCI DSS", "GDPR", "CCPA", "FISMA", "GRC",
            "compliance", "governance", "audit", "risk assessment",
            "control framework", "regulatory", "CISO", "ATO", "authorization",
        ],
    },
    "infosec": {
        "text": (
            "CVE, CVSS, vulnerability, exploit, zero-day, malware, ransomware, phishing, "
            "SOC, SIEM, EDR, XDR, MITRE ATT&CK, threat intelligence, incident response, "
            "penetration testing, red team, blue team, OSINT, IOC, indicators of compromise"
        ),
        "keywords": [
            "CVE", "CVSS", "exploit", "vulnerability", "malware", "ransomware",
            "phishing", "SIEM", "EDR", "XDR", "MITRE", "ATT&CK",
            "red team", "blue team", "pentest", "penetration testing",
            "reverse engineering", "binary exploitation", "CTF",
            "zero-day", "0day", "backdoor", "payload", "C2",
            "threat hunting", "incident response", "OSINT", "IOC",
        ],
    },
}


def resolve_prompt(name_or_string):
    """Return prompt text. Known profile -> its text field. Unknown key -> treat as custom string."""
    profile = PROMPTS.get(name_or_string)
    if profile is not None:
        return profile["text"]
    return name_or_string
