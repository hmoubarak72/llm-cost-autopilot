"""Task-type detection, shared by the verifier (to pick a quality check) and
the offline simulator (to shape fake answers)."""
from __future__ import annotations

import re

Message = dict  # {"role": ..., "content": ...}

_EXTRACT = re.compile(r"\b(extract|pull out|list (all|every|the)|find (all|every|the)|"
                      r"what (is|are) the .{0,30}(name|date|amount|email|phone|address)|"
                      r"return .{0,20}json|parse|fields?)\b", re.I)
_CLASSIFY = re.compile(r"\b(classify|categori[sz]e|label|which category|is (this|it|the) .{0,40}"
                       r"(positive|negative|spam|urgent|bug)|sentiment|tag (this|it)|"
                       r"one of:|choose (one|from))\b", re.I)
_SUMMARIZE = re.compile(r"\b(summari[sz]e|summary|tl;?dr|condense|key points|recap|"
                        r"in (one|two|three|\d+) (sentence|bullet|line)s?)\b", re.I)


def as_text(prompt: str | list[Message]) -> str:
    if isinstance(prompt, str):
        return prompt
    return "\n".join(str(m.get("content", "")) for m in prompt)


def task_type(prompt: str | list[Message]) -> str:
    text = as_text(prompt)
    head = text[:600]  # the instruction usually lives up front
    if _CLASSIFY.search(head):
        return "classification"
    if _EXTRACT.search(head):
        return "extraction"
    if _SUMMARIZE.search(head):
        return "summarization"
    return "general"
