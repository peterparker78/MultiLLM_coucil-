"""Briefing ingestion lives in council.brief (both pipelines use it); this
module re-exports it so existing imports keep working."""

from council.brief import brief_to_text, combine_briefs, text_from_bytes  # noqa: F401
