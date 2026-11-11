"""Compatibility entry point; report generation lives in reporting.summary."""

from reporting.summary import load, paired_bootstrap, summarize

__all__ = ["load", "paired_bootstrap", "summarize"]
