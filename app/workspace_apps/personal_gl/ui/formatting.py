"""Formatting helpers shared across UI modules."""

from __future__ import annotations


def format_money(amount: float) -> str:
    """Format a dollar amount in accounting style."""
    if amount < 0:
        return f"$({abs(amount):,.2f})"
    return f"${amount:,.2f}"


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def format_ratio(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}x"
