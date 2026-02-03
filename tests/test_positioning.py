"""Tests for element positioning functionality."""

import tempfile
from pathlib import Path

import pytest

from stella_mcp.xmile import StellaModel, parse_stmx


class TestUserSpecifiedPositions:
    """Tests for Phase 1: User-specified positions."""

    def test_user_specified_stock_position_preserved(self):
        """User-specified positions should not be overwritten."""
        model = StellaModel("Test")
        model.add_stock("Population", "100", x=400, y=500)
        xml = model.to_xml()

        assert 'x="400"' in xml
        assert 'y="500"' in xml

    def test_user_specified_flow_position_preserved(self):
        """User-specified flow positions should be preserved."""
        model = StellaModel("Test")
        model.add_stock("A", "100", x=200, y=300)
        model.add_stock("B", "100", x=400, y=300)
        model.add_flow("transfer", "10", from_stock="A", to_stock="B", x=350, y=250)
        xml = model.to_xml()

        # Flow position should be preserved
        assert 'x="350"' in xml or 'x="350.0"' in xml

    def test_user_specified_aux_position_preserved(self):
        """User-specified aux positions should be preserved."""
        model = StellaModel("Test")
        model.add_aux("rate", "0.05", x=500, y=100)
        xml = model.to_xml()

        assert 'x="500"' in xml
        assert 'y="100"' in xml

    def test_unspecified_position_auto_laid_out(self):
        """Elements without positions should be auto-positioned."""
        model = StellaModel("Test")
        model.add_stock("Population", "100")  # No x, y
        xml = model.to_xml()

        # Should have default position from _auto_layout
        assert 'x="200"' in xml  # start_x
        assert 'y="300"' in xml  # stock_y

    def test_mixed_positioning_stocks(self):
        """Mix of positioned and unpositioned stocks."""
        model = StellaModel("Test")
        model.add_stock("A", "100", x=400, y=350)  # User positioned
        model.add_stock("B", "100")  # Should get auto-positioned

        # Call _auto_layout and verify positions directly
        model._auto_layout()

        # A should keep its position (400, 350)
        assert model.stocks["A"].x == 400
        assert model.stocks["A"].y == 350

        # B should get auto-positioned at start_x=200, stock_y=300
        assert model.stocks["B"].x == 200
        assert model.stocks["B"].y == 300

    def test_position_zero_is_valid(self):
        """User should be able to position at (0, 0)."""
        model = StellaModel("Test")
        model.add_stock("Origin", "100", x=0, y=0)

        # Call _auto_layout and verify position is preserved
        model._auto_layout()

        # Position (0, 0) should be preserved, not overwritten
        assert model.stocks["Origin"].x == 0
        assert model.stocks["Origin"].y == 0

    def test_flow_points_recalculated_for_user_positioned_stocks(self):
        """Flow points should connect to stocks at their actual positions."""
        model = StellaModel("Test")
        model.add_stock("A", "100", x=100, y=300)
        model.add_stock("B", "100", x=500, y=300)
        model.add_flow("transfer", "10", from_stock="A", to_stock="B")
        xml = model.to_xml()

        # Flow points should span from A to B
        # A.x + 22.5 = 122.5
        # B.x - 22.5 = 477.5
        assert 'x="122.5"' in xml
        assert 'x="477.5"' in xml


class TestRoundTrip:
    """Tests for position preservation on load/save."""

    def test_round_trip_preserves_stock_positions(self):
        """Loading and saving should preserve stock positions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.stmx"

            # Create and save a model
            model1 = StellaModel("Test")
            model1.add_stock("Pop", "100", x=400, y=350)
            filepath.write_text(model1.to_xml())

            # Load and check positions
            model2 = parse_stmx(str(filepath))
            assert model2.stocks["Pop"].x == 400
            assert model2.stocks["Pop"].y == 350

    def test_round_trip_preserves_aux_positions(self):
        """Loading and saving should preserve aux positions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.stmx"

            # Create and save a model
            model1 = StellaModel("Test")
            model1.add_aux("rate", "0.05", x=300, y=100)
            filepath.write_text(model1.to_xml())

            # Load and check positions
            model2 = parse_stmx(str(filepath))
            assert model2.auxs["rate"].x == 300
            assert model2.auxs["rate"].y == 100

    def test_round_trip_preserves_flow_positions(self):
        """Loading and saving should preserve flow positions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.stmx"

            # Create and save a model
            model1 = StellaModel("Test")
            model1.add_stock("A", "100", x=200, y=300)
            model1.add_stock("B", "100", x=400, y=300)
            model1.add_flow("transfer", "10", from_stock="A", to_stock="B")
            filepath.write_text(model1.to_xml())

            # Load and check that flow has position
            model2 = parse_stmx(str(filepath))
            assert model2.flows["transfer"].x is not None
            assert model2.flows["transfer"].y is not None


class TestSmartLayout:
    """Tests for Phase 2: Smart aux layout."""

    def test_constants_grouped_top_left(self):
        """Constants should cluster in top-left area."""
        model = StellaModel("Test")
        model.add_stock("Population", "100")
        model.add_aux("growth_rate", "0.02")  # Constant - no refs
        model.add_aux("capacity", "1000")  # Constant - no refs
        model.add_aux("threshold", "500")  # Constant - no refs

        model._auto_layout()

        # Constants should be at aux_y - 60 = 90
        assert model.auxs["growth_rate"].y == 90
        assert model.auxs["capacity"].y == 90
        assert model.auxs["threshold"].y == 90

        # Should be spaced horizontally
        assert model.auxs["growth_rate"].x == 200  # start_x
        assert model.auxs["capacity"].x == 280  # start_x + aux_spacing
        assert model.auxs["threshold"].x == 360  # start_x + 2*aux_spacing

    def test_flow_modifier_near_flow(self):
        """Auxs used by flows should be positioned near those flows."""
        model = StellaModel("Test")
        model.add_stock("Population", "100")
        # birth_rate is a constant (0.02) so it will be treated as a constant
        # To test flow modifier positioning, the aux must be referenced by a flow
        # AND be an aux that's used in the flow equation
        model.add_aux("birth_rate", "0.02")  # Constant
        model.add_flow("births", "Population * birth_rate", to_stock="Population")
        model.add_connector("birth_rate", "births")

        model._auto_layout()

        # Since birth_rate is used in the flow equation (births), it should be
        # placed near the flow as a "flow modifier"
        flow_x = model.flows["births"].x
        aux_x = model.auxs["birth_rate"].x

        # The aux should be positioned near the flow
        # flow_modifiers are placed at flow.x + offset, where offset is ((i % 3) - 1) * aux_spacing
        # For i=0, offset = -80, so aux_x = flow_x - 80
        assert flow_x is not None
        assert aux_x is not None
        assert abs(aux_x - flow_x) <= 80  # Within one aux_spacing

    def test_stock_dependent_aux_below_stock(self):
        """Auxs that reference stocks should be positioned below them."""
        model = StellaModel("Test")
        model.add_stock("Population", "100", x=300, y=300)
        model.add_aux("pop_indicator", "Population * 0.1")  # References stock

        model._auto_layout()

        # pop_indicator should be below Population (stock_y + 80 = 380)
        assert model.auxs["pop_indicator"].y == 380
        # Should be horizontally aligned with the stock
        assert model.auxs["pop_indicator"].x == 300

    def test_intermediates_in_middle_row(self):
        """Auxs that only reference other auxs should go in the middle row."""
        model = StellaModel("Test")
        model.add_stock("Population", "100")
        model.add_aux("base_rate", "0.02")  # Constant
        model.add_aux("adjusted_rate", "base_rate * 1.5")  # References aux only

        model._auto_layout()

        # base_rate is a constant -> top row (y=90)
        assert model.auxs["base_rate"].y == 90

        # adjusted_rate references only auxs -> intermediate row (y=150)
        assert model.auxs["adjusted_rate"].y == 150

    def test_mixed_categories(self):
        """Test a model with auxs in all categories."""
        model = StellaModel("Test")

        # Stock
        model.add_stock("Population", "100")

        # Constant (no refs)
        model.add_aux("birth_rate", "0.02")

        # Flow that uses birth_rate
        model.add_flow("births", "Population * birth_rate", to_stock="Population")

        # Stock-dependent aux
        model.add_aux("pop_fraction", "Population / 1000")

        # Intermediate (refs aux only)
        model.add_aux("adjusted_rate", "birth_rate * 2")

        model._auto_layout()

        # birth_rate is used by flow -> near flow (aux_y = 150)
        assert model.auxs["birth_rate"].y == 150

        # pop_fraction references stock -> below stock (stock_y + 80 = 380)
        assert model.auxs["pop_fraction"].y == 380

        # adjusted_rate refs only aux -> intermediate (aux_y = 150)
        assert model.auxs["adjusted_rate"].y == 150


class TestExtractVariableRefs:
    """Tests for the _extract_variable_refs helper."""

    def test_simple_equation(self):
        """Extract refs from a simple equation."""
        model = StellaModel("Test")
        refs = model._extract_variable_refs("Population * 0.02")
        assert "Population" in refs
        assert len(refs) == 1

    def test_multiple_refs(self):
        """Extract multiple variable references."""
        model = StellaModel("Test")
        refs = model._extract_variable_refs("Stock1 + Stock2 - Flow1")
        assert "Stock1" in refs
        assert "Stock2" in refs
        assert "Flow1" in refs

    def test_filters_functions(self):
        """Stella functions should be filtered out."""
        model = StellaModel("Test")
        refs = model._extract_variable_refs("MAX(Population, 0)")
        assert "Population" in refs
        assert "MAX" not in refs

    def test_filters_if_then_else(self):
        """IF/THEN/ELSE keywords should be filtered."""
        model = StellaModel("Test")
        refs = model._extract_variable_refs("IF Population > 100 THEN rate ELSE 0")
        assert "Population" in refs
        assert "rate" in refs
        assert "IF" not in refs
        assert "THEN" not in refs
        assert "ELSE" not in refs

    def test_empty_equation(self):
        """Empty equation returns empty set."""
        model = StellaModel("Test")
        refs = model._extract_variable_refs("")
        assert refs == set()

    def test_constant_value(self):
        """Pure numeric value returns empty set."""
        model = StellaModel("Test")
        refs = model._extract_variable_refs("0.02")
        assert refs == set()

    def test_handles_underscores(self):
        """Variable names with underscores are extracted correctly."""
        model = StellaModel("Test")
        refs = model._extract_variable_refs("birth_rate + death_rate")
        assert "birth_rate" in refs
        assert "death_rate" in refs
