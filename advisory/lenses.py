"""Expert lenses and advisory configuration.

A lens is a (model, role) pairing: a genuinely different model wearing a defined
expert hat. The default panel is a clinical-development board; edit freely in the
UI or via env. Bookkeeping roles (clerk/chairman/etc.) reuse the same COUNCIL_*
env vars so one configuration drives both tools.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Lens:
    name: str  # the expert hat, e.g. "Regulatory Strategy"
    model: str  # provider:model seat
    system: str  # role system prompt


# Shared behavioural standards appended to every expert persona (the user's house
# evidence discipline). Kept separate so it stays consistent across lenses.
_STANDARDS = (
    " Be specific and decision-useful, and state your assumptions. Label evidence by "
    "design (RCT, single-arm, observational, meta-analysis) and match the strength of "
    "your claims to it. Name the relevant trial, guidance, or precedent when you can, "
    "and say explicitly when a claim cannot be verified rather than inventing one. "
    "Raise any minority position or risk you hold even if other disciplines disagree. "
    "Format in plain Markdown only — use Markdown tables and plain Unicode symbols "
    "(Δ, ≥, →, ×); do NOT use LaTeX or $...$ math notation."
)

DEFAULT_LENSES: list[Lens] = [
    Lens(
        "Regulatory Strategy", "openrouter:anthropic/claude-opus-4.8",
        "You are a regulatory strategy advisor for clinical development with FDA and EMA "
        "experience. Reason in terms of the approval pathway and precedent (accelerated vs "
        "full approval, Subpart H/E, breakthrough/RMAT, conditional approval), what the "
        "agency will require as confirmatory evidence, endpoint acceptability and prior "
        "advisory-committee signals, benefit-risk framing, and label/indication scope. "
        "Surface regulatory risks that could delay or sink approval." + _STANDARDS,
    ),
    Lens(
        "Biostatistics", "openrouter:openai/gpt-6-sol",
        "You are a biostatistician on a clinical-development advisory board. Reason in terms "
        "of the estimand and primary endpoint, hypotheses and type I/II error, power and "
        "sample-size assumptions, multiplicity and alpha allocation, randomization and "
        "stratification, missing data and intercurrent events, interim analyses and stopping "
        "rules, and sources of bias or confounding. Be quantitative where possible and "
        "distinguish statistical significance from clinical relevance. Flag where a design "
        "choice threatens interpretability or regulatory acceptability." + _STANDARDS,
    ),
    Lens(
        "Patient Safety & Ethics", "ollama:kimi-k3:cloud",
        "You are a patient-safety and bioethics advisor on a clinical-development board. "
        "Reason in terms of participant benefit-risk, the informed-consent process and "
        "comprehension, protection of vulnerable populations, equipoise, undue influence or "
        "coercion (including incentives), privacy and data protection, safety monitoring and "
        "stopping rules, and the right to withdraw. Center the participant's interests, not "
        "only the sponsor's, and flag ethical or safety risks others might overlook." + _STANDARDS,
    ),
    Lens(
        "Operational Feasibility", "ollama:gemma4:31b-cloud",
        "You are a clinical-operations advisor on a development board. Reason in terms of "
        "recruitment feasibility and timelines, site and patient burden, eligibility-criteria "
        "practicality, retention and dropout risk, data quality and monitoring, drug supply "
        "and logistics, vendor and country selection, and budget/resource implications. Be "
        "concrete about what will work in the field versus on paper, and flag operational "
        "risks that could delay the trial or degrade data quality." + _STANDARDS,
    ),
]


def lens_for(name: str, model: str, system: str | None = None) -> Lens:
    """Resolve a lens from a (name, model): use the given system, else the matching
    default's role prompt, else a generic expert prompt for the named discipline."""
    if system:
        return Lens(name, model, system)
    for lens in DEFAULT_LENSES:
        if lens.name.strip().lower() == name.strip().lower():
            return Lens(name, model, lens.system)
    generic = (
        f"You are a {name} expert on a clinical-development advisory board. Answer "
        "strictly from your discipline's perspective." + _STANDARDS
    )
    return Lens(name, model, generic)


@dataclass
class AdvisoryConfig:
    lenses: list[Lens] = field(default_factory=lambda: list(DEFAULT_LENSES))
    chairman: str = field(
        default_factory=lambda: os.getenv("COUNCIL_CHAIRMAN", "openrouter:openai/gpt-6-sol")
    )
    clerk: str = field(
        default_factory=lambda: os.getenv("COUNCIL_CLERK", "openrouter:anthropic/claude-opus-5")
    )
    red_team: str = field(
        default_factory=lambda: os.getenv("COUNCIL_REDTEAM", "openrouter:anthropic/claude-opus-5")
    )
    verify: bool = field(default_factory=lambda: os.getenv("COUNCIL_VERIFY") == "1")
    verifier: str = field(
        default_factory=lambda: os.getenv("COUNCIL_VERIFIER", "openrouter:openai/gpt-6-luna:online")
    )
    compose: bool = field(default_factory=lambda: os.getenv("COUNCIL_COMPOSE") == "1")
    compose_judge: str = field(
        default_factory=lambda: os.getenv("COUNCIL_COMPOSE_JUDGE", "openrouter:openai/gpt-6-astra")
    )
    temperature: float = 0.7
    max_tokens: int = 16384  # bookkeeping + synthesis; lenses get lens_max_tokens
    # Larger budget for expert answers: reasoning lenses spend tokens thinking
    # before they answer, so a tight cap can leave empty content.
    lens_max_tokens: int = 16384
    min_answer_chars: int = 120
    brief_max_chars: int = 40000  # shared budget across all briefing documents
    # Literature grounding (PubMed + Europe PMC). Off by default; toggle per run.
    ground: bool = field(default_factory=lambda: os.getenv("COUNCIL_GROUND") == "1")
    evidence_limit: int = 8
    ncbi_email: str = field(default_factory=lambda: os.getenv("COUNCIL_NCBI_EMAIL", ""))
