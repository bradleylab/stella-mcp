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
        # Connect them so they're in the same subsystem
        model.add_flow("transfer", "10", from_stock="A", to_stock="B")

        # Call _auto_layout and verify positions directly
        model._auto_layout()

        # A should keep its position (400, 350)
        assert model.stocks["A"].x == 400
        assert model.stocks["A"].y == 350

        # B should get auto-positioned (after A in the chain)
        assert model.stocks["B"].x is not None
        assert model.stocks["B"].y == 300  # stock_y

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
    """Tests for Phase 2: Graph-based smart layout using connectors."""

    def test_aux_near_flow_via_connector(self):
        """Auxs connected to flows should be positioned near those flows."""
        model = StellaModel("Test")
        model.add_stock("Population", "100")
        model.add_aux("birth_rate", "0.02")
        model.add_flow("births", "Population * birth_rate", to_stock="Population")
        # Key: add connector to establish relationship
        model.add_connector("birth_rate", "births")

        model._auto_layout()

        # birth_rate should be near the births flow
        flow_x = model.flows["births"].x
        aux_x = model.auxs["birth_rate"].x

        assert flow_x is not None
        assert aux_x is not None
        # Aux should be positioned at or near the flow's x coordinate
        assert abs(aux_x - flow_x) <= 10  # Very close since it's the target

    def test_multiple_auxs_connected_to_same_flow(self):
        """Multiple auxs connected to the same flow cluster near it."""
        model = StellaModel("Test")
        model.add_stock("Population", "100")
        model.add_aux("rate1", "0.02")
        model.add_aux("rate2", "0.03")
        model.add_aux("rate3", "0.01")
        model.add_flow("growth", "Population * rate1 * rate2 * rate3", to_stock="Population")
        model.add_connector("rate1", "growth")
        model.add_connector("rate2", "growth")
        model.add_connector("rate3", "growth")

        model._auto_layout()

        flow_x = model.flows["growth"].x
        assert flow_x is not None

        # All auxs should be near the flow
        for aux_name in ["rate1", "rate2", "rate3"]:
            aux_x = model.auxs[aux_name].x
            assert aux_x is not None
            assert abs(aux_x - flow_x) <= 10

    def test_subsystem_separation(self):
        """Independent subsystems should be visually separated."""
        model = StellaModel("Test")

        # Main subsystem: Population with birth flow
        model.add_stock("Population", "100")
        model.add_aux("birth_rate", "0.02")
        model.add_flow("births", "Population * birth_rate", to_stock="Population")
        model.add_connector("birth_rate", "births")

        # Separate subsystem: Error calculation (no connection to main)
        model.add_aux("Observed", "500")
        model.add_aux("Error", "Observed - 400")
        model.add_connector("Observed", "Error")

        model._auto_layout()

        # Main subsystem elements
        main_x = model.stocks["Population"].x
        # Separate subsystem should be offset to the right
        error_x = model.auxs["Error"].x

        assert main_x is not None
        assert error_x is not None
        # Error subsystem should be to the right of main subsystem
        assert error_x > main_x + 200  # At least subsystem gap apart

    def test_stock_chain_horizontal(self):
        """Stocks connected by flows should be arranged horizontally."""
        model = StellaModel("Test")
        model.add_stock("A", "100")
        model.add_stock("B", "50")
        model.add_stock("C", "25")
        model.add_flow("f1", "10", from_stock="A", to_stock="B")
        model.add_flow("f2", "5", from_stock="B", to_stock="C")

        model._auto_layout()

        # All stocks should be at the same y level
        assert model.stocks["A"].y == model.stocks["B"].y == model.stocks["C"].y == 300

        # Stocks should be arranged left to right following flow direction
        assert model.stocks["A"].x < model.stocks["B"].x < model.stocks["C"].x

    def test_orphan_aux_positioned(self):
        """Auxs with no connectors should still be positioned."""
        model = StellaModel("Test")
        model.add_stock("Population", "100")
        model.add_aux("orphan_param", "42")  # No connector

        model._auto_layout()

        # Orphan should be positioned (not None)
        assert model.auxs["orphan_param"].x is not None
        assert model.auxs["orphan_param"].y is not None
        # Should be at aux_y - 60 = 90 (orphan row)
        assert model.auxs["orphan_param"].y == 90

    def test_aux_without_connector_is_separate_subsystem(self):
        """Auxs without connectors are treated as separate subsystems."""
        model = StellaModel("Test")
        model.add_stock("Population", "100")
        model.add_aux("birth_rate", "0.02")  # No connector
        model.add_flow("births", "Population * birth_rate", to_stock="Population")
        # Note: birth_rate is in flow equation but no connector added

        model._auto_layout()

        # Aux should be positioned (not None)
        assert model.auxs["birth_rate"].x is not None
        assert model.auxs["birth_rate"].y is not None

        # Without a connector, birth_rate is in a separate subsystem
        # and will be placed to the right of the main subsystem
        main_subsystem_max_x = model.stocks["Population"].x
        assert main_subsystem_max_x is not None
        # birth_rate should be offset as a separate subsystem
        assert model.auxs["birth_rate"].x > main_subsystem_max_x


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
