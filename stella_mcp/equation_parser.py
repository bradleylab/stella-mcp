"""Shared equation parsing helpers for Stella-style expressions."""

import re

# Reserved tokens that must not be treated as variable references.
#
# Sources (each token verified against one of these; do not add tokens
# from memory):
# - XMILE v1.0 OASIS spec section 3.5 "Built-In Functions" plus reserved
#   keywords/constants:
#   https://docs.oasis-open.org/xmile/xmile/v1.0/os/xmile-v1.0-os.html
# - isee systems Stella builtin reference (Stella-specific extensions):
#   https://iseesystems.com/resources/help/v2/Content/08-Reference/07-Builtins/Overview_Builtins.htm

# XMILE v1.0 spec: operators/keywords, constants, and built-in functions.
_XMILE_SPEC_TOKENS = {
    # Keywords and operators
    "IF", "THEN", "ELSE", "AND", "OR", "NOT", "MOD",
    # Constants
    "PI", "INF",
    # Mathematical
    "ABS", "ARCCOS", "ARCSIN", "ARCTAN", "COS", "EXP", "INT",
    "LN", "LOG10", "MAX", "MIN", "SIN", "SQRT", "TAN",
    # Statistical
    "EXPRND", "LOGNORMAL", "NORMAL", "POISSON", "RANDOM",
    # Delay
    "DELAY", "DELAY1", "DELAY3", "DELAYN", "FORCST",
    "SMTH1", "SMTH3", "SMTHN", "TREND",
    # Test input
    "PULSE", "RAMP", "STEP",
    # Time
    "DT", "STARTTIME", "STOPTIME", "TIME",
    # Miscellaneous
    "INIT", "PREVIOUS", "SELF",
}

# Stella builtins beyond the XMILE spec (isee builtin reference, v2).
_STELLA_EXTENSION_TOKENS = {
    "ALLOCATE", "ATTRCOUNT", "ATTRMAX", "ATTRMEAN", "ATTRMIN",
    "ATTRSTDDEV", "BETA", "BINOMIAL", "CAPACITY", "CGROWTH",
    "CLOCKTIME", "COMBINATIONS", "COSWAVE", "COUNTER", "CTFLOW",
    "CTMAX", "CTMEAN", "CTMIN", "CTSTDDEV", "CYCLETIME", "DERIVN",
    "ENDVAL", "FACTORIAL", "FV", "GAMMA", "GAMMALN", "GEOMETRIC",
    "HISTORY", "INTERPOLATE", "INVNORM", "LOGISTIC", "LOOKUP",
    "LOOKUPAREA", "LOOKUPINV", "LOOKUPMEAN", "MEAN", "MLPANN",
    "MONTECARLO", "NAN", "NEGBINOMIAL", "NORMALCDF", "NPV", "OSTATE",
    "PARETO", "PERCENT", "PERMUTATIONS", "PMT", "PROCTIME", "PROD",
    "PV", "RANK", "REWORK", "ROOTN", "ROUND", "RUNCOUNT",
    "SENSIRUNCOUNT", "SINWAVE", "SIZE", "STDDEV", "SUM", "THROUGHPUT",
    "TRIANGULAR", "UNIFORM", "WEIBULL",
}

# Tokens carried over from earlier releases of this package that are not in
# either reference above (Vensim-style aliases and literals this parser has
# always treated as reserved, e.g. GRAPH(...) emitted for graphical
# functions). Kept for backward compatibility with existing models.
_LEGACY_COMPAT_TOKENS = {
    "LOG", "SMOOTH", "SMOOTH3", "SMOOTHN", "GRAPH", "SAFEDIV", "IRR",
    "TRUE", "FALSE", "E",
}

STELLA_RESERVED_TOKENS = (
    _XMILE_SPEC_TOKENS | _STELLA_EXTENSION_TOKENS | _LEGACY_COMPAT_TOKENS
)

_TOKEN_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_QUOTED_SPAN_PATTERN = re.compile(r'"([^"]*)"')


def _is_reserved_or_numeric(token: str) -> bool:
    if token.upper() in STELLA_RESERVED_TOKENS:
        return True
    try:
        float(token)
        return True
    except ValueError:
        return False


def extract_quoted_references(equation: str) -> set[str]:
    """Extract quoted identifiers from an equation.

    XMILE allows variables whose names contain spaces or other specials to
    be referenced in quoted form (e.g. ``"net growth rate" * Population``).
    Returns the inner text of each non-empty quoted span, quotes stripped,
    case preserved. A quoted span may also be a genuine string argument
    (e.g. a label) — callers decide by checking membership against model
    variables.
    """
    if not equation:
        return set()
    return {
        span
        for span in _QUOTED_SPAN_PATTERN.findall(equation)
        if span.strip() and not _is_reserved_or_numeric(span)
    }


def extract_variable_references(equation: str) -> set[str]:
    """Extract variable-like identifiers from an equation.

    Returns identifiers exactly as they appear (case preserved), including
    the inner text of quoted identifiers. Reserved Stella tokens and
    numeric literals are filtered out.
    """
    if not equation:
        return set()

    refs = extract_quoted_references(equation)

    # Remove quoted spans before token-scanning so their contents are not
    # re-tokenized word by word.
    equation = _QUOTED_SPAN_PATTERN.sub("", equation)

    for token in _TOKEN_PATTERN.findall(equation):
        if not _is_reserved_or_numeric(token):
            refs.add(token)
    return refs
