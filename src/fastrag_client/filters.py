"""Filter builder for fastrag query API.

Usage:
    from fastrag_client.filters import F

    expr = (F.severity == "HIGH") & (F.cvss >= 7.0)
    str(expr)  # "severity = HIGH, cvss >= 7.0"
"""

from __future__ import annotations

from typing import Any


class FilterExpr:
    """A single filter condition or a conjunction of conditions."""

    def __init__(self, parts: list[str]) -> None:
        self._parts = parts

    def __and__(self, other: FilterExpr) -> FilterExpr:
        return FilterExpr(self._parts + other._parts)

    def __str__(self) -> str:
        return ", ".join(self._parts)

    def __repr__(self) -> str:
        return f"FilterExpr({self._parts!r})"


def _format_value(value: Any) -> str:
    return str(value)


class FieldExpr:
    """Proxy for a field name. Comparison operators produce FilterExpr."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __eq__(self, other: object) -> FilterExpr:  # type: ignore[override]
        return FilterExpr([f"{self._name} = {_format_value(other)}"])

    def __ge__(self, other: object) -> FilterExpr:  # type: ignore[override]
        return FilterExpr([f"{self._name} >= {_format_value(other)}"])

    def __gt__(self, other: object) -> FilterExpr:  # type: ignore[override]
        return FilterExpr([f"{self._name} > {_format_value(other)}"])

    def __le__(self, other: object) -> FilterExpr:  # type: ignore[override]
        return FilterExpr([f"{self._name} <= {_format_value(other)}"])

    def __lt__(self, other: object) -> FilterExpr:  # type: ignore[override]
        return FilterExpr([f"{self._name} < {_format_value(other)}"])

    def in_(self, values: list[Any]) -> FilterExpr:
        formatted = ", ".join(_format_value(v) for v in values)
        return FilterExpr([f"{self._name} IN ({formatted})"])


class FieldFactory:
    """Factory that creates FieldExpr via attribute access.

    Usage: F.severity returns FieldExpr("severity")
    """

    def __getattr__(self, name: str) -> FieldExpr:
        return FieldExpr(name)


F = FieldFactory()
