"""Shared equation parsing helpers for Stella-style expressions."""

import re

# Stella/XMILE built-in functions and reserved constants/keywords.
STELLA_RESERVED_TOKENS = {
    "IF", "THEN", "ELSE", "AND", "OR", "NOT",
    "MIN", "MAX", "ABS", "SIN", "COS", "TAN",
    "EXP", "LN", "LOG", "LOG10", "SQRT", "INT",
    "ROUND", "MOD", "TIME", "DT", "STARTTIME", "STOPTIME",
    "DELAY", "DELAY1", "DELAY3", "DELAYN",
    "SMOOTH", "SMOOTH3", "SMOOTHN", "SMTH1", "SMTH3", "SMTHN",
    "TREND", "FORCST", "PULSE", "STEP", "RAMP",
    "RANDOM", "NORMAL", "POISSON", "EXPRND",
    "PREVIOUS", "INIT", "SELF", "SUM", "MEAN",
    "GRAPH", "LOOKUP", "INTERPOLATE", "HISTORY",
    "SAFEDIV", "NPV", "IRR", "COUNTER",
    "TRUE", "FALSE", "PI", "E", "INF", "NAN",
}

_TOKEN_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_STRING_LITERAL_PATTERN = re.compile(r'"[^"]*"')


def extract_variable_references(equation: str) -> set[str]:
    """Extract variable-like identifiers from an equation.

    Returns identifiers exactly as they appear (case preserved).
    Reserved Stella tokens and numeric literals are filtered out.
    """
    if not equation:
        return set()

    # Ignore quoted string literals so labels in functions (e.g., LOOKUP labels)
    # do not get interpreted as variable references.
    equation = _STRING_LITERAL_PATTERN.sub("", equation)

    refs: set[str] = set()
    for token in _TOKEN_PATTERN.findall(equation):
        if token.upper() in STELLA_RESERVED_TOKENS:
            continue
        try:
            float(token)
            continue
        except ValueError:
            refs.add(token)
    return refs
