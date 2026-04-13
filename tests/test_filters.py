from fastrag_client.filters import F


def test_eq_string():
    expr = F.severity == "HIGH"
    assert str(expr) == "severity = HIGH"


def test_eq_numeric():
    expr = F.cvss == 9.8
    assert str(expr) == "cvss = 9.8"


def test_ge():
    expr = F.cvss >= 7.0
    assert str(expr) == "cvss >= 7.0"


def test_gt():
    expr = F.cvss > 5.0
    assert str(expr) == "cvss > 5.0"


def test_le():
    expr = F.cvss <= 3.0
    assert str(expr) == "cvss <= 3.0"


def test_lt():
    expr = F.cvss < 3.0
    assert str(expr) == "cvss < 3.0"


def test_in():
    expr = F.severity.in_(["HIGH", "CRITICAL"])
    assert str(expr) == "severity IN (HIGH, CRITICAL)"


def test_and_two():
    expr = (F.severity == "HIGH") & (F.cvss >= 7.0)
    assert str(expr) == "severity = HIGH, cvss >= 7.0"


def test_and_three():
    expr = (F.severity == "HIGH") & (F.cvss >= 7.0) & (F.source_tool == "semgrep")
    assert str(expr) == "severity = HIGH, cvss >= 7.0, source_tool = semgrep"


def test_field_access_returns_field_expr():
    field = F.severity
    assert hasattr(field, "in_")
    assert hasattr(field, "__eq__")


def test_in_single_value():
    expr = F.severity.in_(["HIGH"])
    assert str(expr) == "severity IN (HIGH)"
