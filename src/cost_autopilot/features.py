"""Phase 2.2: hand-engineered complexity features.

These are the features the guide asks for (token count, analytical verbs,
constraint count, provided context, output-format complexity) plus a few that
proved useful. The classifier combines them with TF-IDF over the instruction.
"""
from __future__ import annotations

import re

import numpy as np

SIMPLE_VERBS = r"\b(reformat|convert|extract|rewrite|translate|fix (the )?(typo|grammar|spelling)|" \
               r"capitali[sz]e|list|what is|who is|when (is|was|did)|lowercase|uppercase|sort|" \
               r"count|format|copy|rename|pull out|find)\b"
MODERATE_VERBS = r"\b(summari[sz]e|classify|categori[sz]e|explain|describe|outline|compare|" \
                 r"contrast|organi[sz]e|tag|draft|identify|group|review|rank)\b"
COMPLEX_VERBS = r"\b(analy[sz]e|evaluate|design|architect|prove|derive|reason|justify|critique|" \
                r"strategy|strategi[sz]e|trade-?offs?|recommend|plan|forecast|debug|optimi[sz]e|" \
                r"diagnose|negotiate|assess|weigh|invent|write a (story|poem|novel|screenplay|speech)|" \
                r"step[- ]by[- ]step|implications?|second-order|what would happen|why)\b"
CREATIVE = r"\b(story|poem|haiku|lyrics|screenplay|fiction|creative|imagine|character|metaphor|" \
           r"slogan|pitch|persuasive|essay)\b"
CONSTRAINT = r"\b(must|should|do not|don't|never|always|at least|at most|no more than|exactly|" \
             r"only|without|within|limit|under \d+|between \d+ and \d+|ensure|make sure)\b"
FORMAT = r"\b(json|yaml|xml|csv|table|markdown|bullet|numbered|schema|columns?|sections?|" \
         r"headings?|code block|format)\b"
CONTEXT_MARKERS = r"(context:|text:|email:|document:|article:|passage:|input:|data:|\"\"\"|```|" \
                  r"<doc>|---\n|below:|following text|following email|this text)"
MULTI_STEP = r"\b(first|then|next|finally|after that|step \d|\d\)|\d\.\s)"

FEATURE_NAMES = [
    "log_tokens", "n_simple", "n_moderate", "n_complex", "n_creative", "n_constraints",
    "n_format", "has_context", "context_ratio", "n_multistep", "n_questions",
    "n_sentences", "instr_tokens_log",
]


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.I))


def split_instruction(prompt: str) -> tuple[str, str]:
    """Best-effort split into (instruction, provided context)."""
    m = re.search(CONTEXT_MARKERS, prompt, flags=re.I)
    if m and m.start() < len(prompt) - 20:
        return prompt[: m.start()], prompt[m.start():]
    parts = prompt.split("\n\n", 1)
    if len(parts) == 2 and len(parts[1]) > 2 * len(parts[0]) and len(parts[1]) > 200:
        return parts[0], parts[1]
    return prompt, ""


def extract_features(prompt: str) -> dict[str, float]:
    instr, ctx = split_instruction(prompt)
    toks = approx_tokens(prompt)
    return {
        "log_tokens": float(np.log1p(toks)),
        "n_simple": _count(SIMPLE_VERBS, instr),
        "n_moderate": _count(MODERATE_VERBS, instr),
        "n_complex": _count(COMPLEX_VERBS, instr),
        "n_creative": _count(CREATIVE, instr),
        "n_constraints": _count(CONSTRAINT, instr),
        "n_format": _count(FORMAT, instr),
        "has_context": float(bool(ctx)),
        "context_ratio": len(ctx) / max(1, len(prompt)),
        "n_multistep": _count(MULTI_STEP, instr),
        "n_questions": instr.count("?"),
        "n_sentences": len(re.findall(r"[.!?](\s|$)", instr)),
        "instr_tokens_log": float(np.log1p(approx_tokens(instr))),
    }


def feature_vector(prompt: str) -> list[float]:
    f = extract_features(prompt)
    return [f[n] for n in FEATURE_NAMES]


def heuristic_tier(prompt: str) -> int:
    """Rule-of-thumb tier used by the simulator for unseen prompts and as a
    fallback if no trained classifier exists."""
    f = extract_features(prompt)
    score = f["n_complex"] * 2 + f["n_creative"] * 1.5 + f["n_multistep"] * 0.5 \
        + f["n_constraints"] * 0.3 + f["n_moderate"] * 0.8 - f["n_simple"] * 0.8
    if score >= 2.5:
        return 3
    if score >= 0.8:
        return 2
    return 1
