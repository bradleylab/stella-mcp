"""Model validation for Stella system dynamics models."""

from dataclasses import dataclass

from .equation_parser import (
    extract_quoted_references,
    extract_variable_references,
    is_stella_reserved_identifier,
)
from .xmile import StellaModel

# Canonical singular form for the XMILE-conventional time units (XMILE v1.0
# sim_specs time_units is a free string, but these are the calendar units used
# in practice). Used to collapse "years" vs "year" when checking that a flow's
# units read as stock-units-per-time-unit. Fixed table, not a stemmer.
_TIME_UNIT_SINGULAR = {
    "years": "year", "year": "year", "yr": "year", "yrs": "year",
    "months": "month", "month": "month", "mo": "month", "mos": "month",
    "weeks": "week", "week": "week", "wk": "week", "wks": "week",
    "days": "day", "day": "day",
    "hours": "hour", "hour": "hour", "hr": "hour", "hrs": "hour",
    "minutes": "minute", "minute": "minute", "min": "minute", "mins": "minute",
    "seconds": "second", "second": "second", "sec": "second", "secs": "second",
}


def _norm_units(units: str) -> str:
    """Conservative unit normalization: lowercase, drop whitespace."""
    return units.strip().lower().replace(" ", "")


def normalize_time_unit(units: str) -> str:
    """Normalize a time unit using the documented calendar-unit mapping."""
    normalized = _norm_units(units)
    return _TIME_UNIT_SINGULAR.get(normalized, normalized)


@dataclass
class ValidationError:
    """Represents a validation error or warning."""
    severity: str  # "error" or "warning"
    category: str  # e.g., "undefined_variable", "mass_balance", "missing_connection"
    message: str
    variable: str | None = None


class ModelValidator:
    """Validates Stella models for common errors."""

    def __init__(self, model: StellaModel):
        self.model = model
        self.errors: list[ValidationError] = []

    def validate(self) -> list[ValidationError]:
        """Run all validation checks and return errors/warnings."""
        self.errors = []

        self._check_reserved_identifiers()
        self._check_undefined_variables()
        self._check_mass_balance()
        self._check_missing_connections()
        self._check_connector_endpoints()
        self._check_orphan_flows()
        self._check_stock_inflow_outflow_consistency()
        self._check_circular_dependencies()
        self._check_modules()
        self._check_units()
        self._check_unused_variables()

        return self.errors

    def _check_reserved_identifiers(self):
        """Warn before Stella silently renames variables that shadow built-ins."""
        variables = [
            *(stock.name for stock in self.model.stocks.values()),
            *(flow.name for flow in self.model.flows.values()),
            *(aux.name for aux in self.model.auxs.values()),
        ]
        for name in sorted(variables, key=lambda value: (value.casefold(), value)):
            if is_stella_reserved_identifier(name):
                self.errors.append(
                    ValidationError(
                        severity="warning",
                        category="reserved_identifier",
                        message=(
                            f"Variable '{name}' conflicts with a Stella/XMILE built-in; "
                            "use a descriptive non-reserved name for stable desktop round-trips"
                        ),
                        variable=self.model._normalize_name(name),
                    )
                )

    def _get_all_variable_names(self) -> set[str]:
        """Get all variable names in the model."""
        names = set()
        for name in self.model.stocks:
            names.add(name)
        for name in self.model.flows:
            names.add(name)
        for name in self.model.auxs:
            names.add(name)
        return names

    def _extract_variable_references(self, equation: str) -> set[str]:
        """Extract variable names referenced in an equation."""
        return extract_variable_references(equation)

    def _check_undefined_variables(self):
        """Check for references to undefined variables."""
        all_vars = self._get_all_variable_names()

        variables = [
            ("Flow", name, flow.name, flow.equation)
            for name, flow in self.model.flows.items()
        ] + [
            ("Auxiliary", name, aux.name, aux.equation)
            for name, aux in self.model.auxs.items()
        ]

        for kind, name, display_name, equation in variables:
            refs = self._extract_variable_references(equation)
            quoted = extract_quoted_references(equation)
            for ref in refs:
                if self.model._normalize_name(ref) in all_vars:
                    continue
                if ref in quoted:
                    # A quoted span that matches no variable may be a genuine
                    # string argument (e.g., a label), so this is not a hard error.
                    self.errors.append(ValidationError(
                        severity="warning",
                        category="unresolved_quoted_reference",
                        message=(
                            f"{kind} '{display_name}' contains quoted reference "
                            f"'\"{ref}\"' that matches no variable (string label, "
                            f"or a typo in a quoted variable name)"
                        ),
                        variable=name
                    ))
                else:
                    self.errors.append(ValidationError(
                        severity="error",
                        category="undefined_variable",
                        message=f"{kind} '{display_name}' references undefined variable '{ref}'",
                        variable=name
                    ))

    def _check_mass_balance(self):
        """Check for potential mass balance issues."""
        # For each stock, check if it has at least one inflow or outflow
        for name, stock in self.model.stocks.items():
            if not stock.inflows and not stock.outflows:
                self.errors.append(ValidationError(
                    severity="warning",
                    category="mass_balance",
                    message=f"Stock '{stock.name}' has no inflows or outflows (isolated reservoir)",
                    variable=name
                ))

        # Check if any flows reference stocks that don't exist
        for name, flow in self.model.flows.items():
            if flow.from_stock and flow.from_stock not in self.model.stocks:
                self.errors.append(ValidationError(
                    severity="error",
                    category="mass_balance",
                    message=f"Flow '{flow.name}' references non-existent from_stock '{flow.from_stock}'",
                    variable=name
                ))
            if flow.to_stock and flow.to_stock not in self.model.stocks:
                self.errors.append(ValidationError(
                    severity="error",
                    category="mass_balance",
                    message=f"Flow '{flow.name}' references non-existent to_stock '{flow.to_stock}'",
                    variable=name
                ))

    def _check_missing_connections(self):
        """Check for missing connectors based on equation references."""
        all_vars = self._get_all_variable_names()

        # Build set of existing connections
        existing_connections = set()
        for conn in self.model.connectors:
            existing_connections.add((conn.from_var, conn.to_var))

        # Check flow equations for missing connectors
        for name, flow in self.model.flows.items():
            refs = self._extract_variable_references(flow.equation)
            for ref in refs:
                normalized = self.model._normalize_name(ref)
                if normalized in all_vars and normalized != name:
                    if (normalized, name) not in existing_connections:
                        self.errors.append(ValidationError(
                            severity="warning",
                            category="missing_connection",
                            message=f"Flow '{flow.name}' uses '{ref}' but no connector exists",
                            variable=name
                        ))

        # Check aux equations for missing connectors
        for name, aux in self.model.auxs.items():
            refs = self._extract_variable_references(aux.equation)
            for ref in refs:
                normalized = self.model._normalize_name(ref)
                if normalized in all_vars and normalized != name:
                    if (normalized, name) not in existing_connections:
                        self.errors.append(ValidationError(
                            severity="warning",
                            category="missing_connection",
                            message=f"Auxiliary '{aux.name}' uses '{ref}' but no connector exists",
                            variable=name
                        ))

    def _check_orphan_flows(self):
        """Check for flows that aren't connected to any stock."""
        for name, flow in self.model.flows.items():
            if not flow.from_stock and not flow.to_stock:
                self.errors.append(ValidationError(
                    severity="warning",
                    category="orphan_flow",
                    message=f"Flow '{flow.name}' is not connected to any stock",
                    variable=name
                ))

    def _check_connector_endpoints(self):
        """Check connectors reference variables that exist in the model."""
        all_vars = self._get_all_variable_names()
        for connector in self.model.connectors:
            if connector.from_var not in all_vars:
                self.errors.append(ValidationError(
                    severity="error",
                    category="connector_endpoint_missing",
                    message=(
                        f"Connector uid={connector.uid} references missing source "
                        f"'{connector.from_var}'"
                    ),
                    variable=connector.to_var,
                ))
            if connector.to_var not in all_vars:
                self.errors.append(ValidationError(
                    severity="error",
                    category="connector_endpoint_missing",
                    message=(
                        f"Connector uid={connector.uid} references missing target "
                        f"'{connector.to_var}'"
                    ),
                    variable=connector.from_var,
                ))

    def _check_stock_inflow_outflow_consistency(self):
        """Check that stock inflows/outflows match flow definitions."""
        for name, stock in self.model.stocks.items():
            # Check inflows
            for inflow in stock.inflows:
                if inflow not in self.model.flows:
                    self.errors.append(ValidationError(
                        severity="error",
                        category="undefined_variable",
                        message=f"Stock '{stock.name}' references undefined inflow '{inflow}'",
                        variable=name
                    ))
                elif self.model.flows[inflow].to_stock != name:
                    self.errors.append(ValidationError(
                        severity="warning",
                        category="inconsistent_flow",
                        message=f"Stock '{stock.name}' lists '{inflow}' as inflow, but flow doesn't point to this stock",
                        variable=name
                    ))

            # Check outflows
            for outflow in stock.outflows:
                if outflow not in self.model.flows:
                    self.errors.append(ValidationError(
                        severity="error",
                        category="undefined_variable",
                        message=f"Stock '{stock.name}' references undefined outflow '{outflow}'",
                        variable=name
                    ))
                elif self.model.flows[outflow].from_stock != name:
                    self.errors.append(ValidationError(
                        severity="warning",
                        category="inconsistent_flow",
                        message=f"Stock '{stock.name}' lists '{outflow}' as outflow, but flow doesn't originate from this stock",
                        variable=name
                    ))

    def _check_circular_dependencies(self):
        """Check for circular dependencies in auxiliary variables (excluding stocks/flows)."""
        # Build dependency graph for aux variables only
        deps: dict[str, set[str]] = {}
        aux_names = set(self.model.auxs.keys())

        for name, aux in self.model.auxs.items():
            refs = self._extract_variable_references(aux.equation)
            # Only track dependencies on other aux variables; normalization
            # maps quoted display-name refs onto internal keys, and unresolved
            # quoted spans drop out of the intersection.
            deps[name] = {self.model._normalize_name(ref) for ref in refs} & aux_names

        # Check for cycles using DFS
        def has_cycle(node: str, visited: set[str], rec_stack: set[str]) -> list[str]:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in deps.get(node, set()):
                if neighbor not in visited:
                    cycle = has_cycle(neighbor, visited, rec_stack)
                    if cycle:
                        return [node] + cycle
                elif neighbor in rec_stack:
                    return [node, neighbor]

            rec_stack.remove(node)
            return []

        visited: set[str] = set()
        for name in aux_names:
            if name not in visited:
                cycle = has_cycle(name, visited, set())
                if cycle:
                    cycle_str = " -> ".join(cycle)
                    self.errors.append(ValidationError(
                        severity="error",
                        category="circular_dependency",
                        message=f"Circular dependency detected among auxiliaries: {cycle_str}",
                        variable=cycle[0]
                    ))
                    break  # Only report first cycle found

    def _check_modules(self):
        """Check module integrity (empty modules and stale members)."""
        all_vars = self._get_all_variable_names()

        for module_key, module in self.model.modules.items():
            if not module.members:
                self.errors.append(ValidationError(
                    severity="warning",
                    category="module_empty",
                    message=f"Module '{module.name}' has no members",
                    variable=module_key,
                ))
                continue

            for member in module.members:
                if member not in all_vars:
                    self.errors.append(ValidationError(
                        severity="error",
                        category="module_member_missing",
                        message=f"Module '{module.name}' references missing member '{member}'",
                        variable=module_key,
                    ))

    def _check_units(self):
        """Conservative, warning-tier unit consistency checks.

        These are heuristics, not dimensional analysis: they only fire when
        near-certain, because a false units warning trains users to ignore
        the validator. When in doubt, stay silent.
        """
        self._check_units_missing()
        self._check_units_inconsistent()

    def _check_units_missing(self):
        """Warn when some stocks/flows carry units but others are blank.

        A fully unitless model (common in teaching) stays silent; mixed
        models are where unit mistakes hide. Auxiliaries are exempt —
        dimensionless parameters are routine.
        """
        stocks_and_flows = [
            ("Stock", name, stock) for name, stock in self.model.stocks.items()
        ] + [
            ("Flow", name, flow) for name, flow in self.model.flows.items()
        ]
        if not any(item.units.strip() for _, _, item in stocks_and_flows):
            return  # fully unitless model
        for kind, name, item in stocks_and_flows:
            if not item.units.strip():
                self.errors.append(ValidationError(
                    severity="warning",
                    category="units_missing",
                    message=(
                        f"{kind} '{item.name}' has no units while other variables "
                        f"in the model do"
                    ),
                    variable=name,
                ))

    def _check_units_inconsistent(self):
        """Warn when a flow's units don't read as stock-units-per-time-unit.

        Only fires when every stock attached to the flow shares the same
        non-empty units (a conversion flow between differently-united stocks
        is legitimate, so those are skipped). Anything the normalizer cannot
        confidently parse stays silent.
        """
        time_unit = self.model.sim_specs.time_units
        for name, flow in self.model.flows.items():
            attached = [
                (s, self.model.stocks[s].name, self.model.stocks[s].units.strip())
                for s in (flow.from_stock, flow.to_stock)
                if s and s in self.model.stocks
            ]
            attached_units = [units for _, _, units in attached]
            # Need at least one attached stock, all with the same non-empty units.
            if not attached_units or not all(attached_units):
                continue
            if len({_norm_units(u) for u in attached_units}) != 1:
                continue  # conversion flow between differing units
            if not flow.units.strip():
                continue  # blank flow units are units_missing, not inconsistent

            stock_display = attached[0][1]
            stock_units = attached_units[0]
            expected = f"{stock_units}/{time_unit}"
            flow_units = flow.units.strip()
            slash_count = flow_units.count("/")
            if slash_count == 1:
                numerator, denominator = flow_units.split("/")
                consistent = (
                    _norm_units(numerator) == _norm_units(stock_units)
                    and normalize_time_unit(denominator) == normalize_time_unit(time_unit)
                )
                if consistent:
                    continue
            elif slash_count == 0:
                # No division operator. We cannot confidently parse rate forms
                # like "GtC yr-1", "people per year", or "molecules s^-1", so
                # the only thing flagged is flow units identical to the stock's
                # (the per-time operator was dropped entirely). Everything else
                # stays silent — a false units warning trains users to ignore
                # the validator.
                if _norm_units(flow_units) != _norm_units(stock_units):
                    continue
            else:
                continue  # multiple slashes: too complex to judge confidently

            self.errors.append(ValidationError(
                severity="warning",
                category="units_inconsistent",
                message=(
                    f"Flow '{flow.name}' has units '{flow_units}' but stock "
                    f"'{stock_display}' ({stock_units}) over time unit "
                    f"'{time_unit}' implies '{expected}'"
                ),
                variable=name,
            ))

    def _check_unused_variables(self):
        """Warn about auxiliaries referenced by no equation or connector.

        Only auxiliaries are flagged: a stock's state is a result, not an
        input, and a flow attached to a stock is doing work even if nothing
        reads it. A connector counts as use (graphical-function inputs arrive
        that way), as does a reference inside a quoted identifier and a
        reference from a stock's initial-value equation.

        Known accepted limitation: a stale connector whose target equation
        doesn't actually use the source will mask this warning, because the
        validator doesn't cross-check connector-vs-equation usage. That is a
        separate concern; tightening it here would duplicate
        _check_missing_connections in reverse.
        """
        used: set[str] = set()
        equations = (
            [flow.equation for flow in self.model.flows.values()]
            + [aux.equation for aux in self.model.auxs.values()]
            + [stock.initial_value for stock in self.model.stocks.values()]
        )
        for equation in equations:
            for ref in self._extract_variable_references(equation):
                used.add(self.model._normalize_name(ref))
        # Connector sources count as use (already normalized internal keys).
        for connector in self.model.connectors:
            used.add(connector.from_var)

        for name, aux in self.model.auxs.items():
            if name not in used:
                self.errors.append(ValidationError(
                    severity="warning",
                    category="unused_variable",
                    message=(
                        f"Auxiliary '{aux.name}' is defined but referenced by no "
                        f"equation or connector"
                    ),
                    variable=name,
                ))


def validate_model(model: StellaModel) -> list[ValidationError]:
    """Convenience function to validate a model."""
    validator = ModelValidator(model)
    return validator.validate()
