"""Tests for module/group support."""

import asyncio

import stella_mcp.server as server_mod
from stella_mcp.validator import validate_model
from stella_mcp.xmile import StellaModel, parse_stmx


def _tool_text(result):
    """Return first text content from either legacy list or CallToolResult responses."""
    if hasattr(result, "content"):
        return result.content[0].text
    return result[0].text


def test_create_module_and_add_members():
    """Model can create modules and add members."""
    model = StellaModel("ModuleTest")
    model.add_stock("S", "100")
    model.add_aux("k", "0.1")

    module = model.create_module("Core", members=["S"])
    assert module.members == ["S"]

    module = model.add_to_module("Core", ["k"])
    assert "k" in module.members


def test_module_round_trip_preserved(tmp_path):
    """Modules should persist through save and parse."""
    filepath = tmp_path / "modules.stmx"
    model = StellaModel("ModuleRoundTrip")
    model.add_stock("Population", "100")
    model.add_aux("rate", "0.1")
    model.create_module("Dynamics", members=["Population", "rate"])
    filepath.write_text(model.to_xml(), encoding="utf-8")

    loaded = parse_stmx(str(filepath))
    assert "Dynamics" in loaded.modules
    members = set(loaded.modules["Dynamics"].members)
    assert members == {"Population", "rate"}


def test_module_view_geometry_round_trip(tmp_path):
    """Module view box geometry should persist through save/load."""
    filepath = tmp_path / "module_view.stmx"
    model = StellaModel("ModuleViewRoundTrip")
    model.add_stock("A", "10", x=100, y=100)
    model.create_module("Boxed", members=["A"])
    model.set_module_view("Boxed", x=200, y=220, width=300, height=160)

    filepath.write_text(model.to_xml(auto_layout=False), encoding="utf-8")
    loaded = parse_stmx(str(filepath))
    assert "Boxed" in loaded.modules
    mod = loaded.modules["Boxed"]
    assert mod.x == 200
    assert mod.y == 220
    assert mod.width == 300
    assert mod.height == 160


def test_module_style_round_trip(tmp_path):
    """Module view style should persist through save/load."""
    filepath = tmp_path / "module_style.stmx"
    model = StellaModel("ModuleStyleRoundTrip")
    model.add_stock("A", "10")
    model.create_module("Styled", members=["A"])
    model.set_module_style(
        "Styled",
        border_color="#666666",
        background="#FFF7E6",
        font_color="#333333",
        font_size="10pt",
        label_side="left",
    )

    filepath.write_text(model.to_xml(auto_layout=False), encoding="utf-8")
    loaded = parse_stmx(str(filepath))
    mod = loaded.modules["Styled"]
    assert mod.border_color == "#666666"
    assert mod.background == "#FFF7E6"
    assert mod.font_color == "#333333"
    assert mod.font_size == "10pt"
    assert mod.label_side == "left"


def test_auto_place_module_boxes_sets_geometry():
    """Auto-placement should set module geometry from member positions."""
    model = StellaModel("AutoModuleBox")
    model.add_stock("S", "100", x=120, y=240)
    model.add_aux("k", "0.1", x=280, y=240)
    model.create_module("Core", members=["S", "k"])

    model.auto_place_module_boxes(padding=20, min_width=100, min_height=80)
    module = model.modules["Core"]
    assert module.x is not None
    assert module.y is not None
    assert module.width is not None and module.width >= 100
    assert module.height is not None and module.height >= 80


def test_server_module_tools(monkeypatch):
    """Server tools can create/list/update modules."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 1111)

    asyncio.run(server_mod.call_tool("create_model", {"name": "ModuleServer", "model_id": "m1"}))
    asyncio.run(
        server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S", "initial_value": "100"})
    )
    asyncio.run(
        server_mod.call_tool("add_aux", {"model_id": "m1", "name": "k", "equation": "0.1"})
    )

    created = asyncio.run(
        server_mod.call_tool(
            "create_module",
            {"model_id": "m1", "name": "Core", "members": ["S"]},
        )
    )
    assert "Created module 'Core'" in _tool_text(created)

    updated = asyncio.run(
        server_mod.call_tool(
            "add_to_module",
            {"model_id": "m1", "module_name": "Core", "members": ["k"]},
        )
    )
    assert "total members: 2" in _tool_text(updated)

    listed = asyncio.run(server_mod.call_tool("list_modules", {"model_id": "m1"}))
    assert "Core: S, k" in _tool_text(listed)


def test_server_set_and_auto_module_view_tools(monkeypatch):
    """Server module view tools should set and auto-place module geometry."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 1212)

    asyncio.run(server_mod.call_tool("create_model", {"name": "ModuleViewServer", "model_id": "m1"}))
    asyncio.run(server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S", "initial_value": "100"}))
    asyncio.run(server_mod.call_tool("create_module", {"model_id": "m1", "name": "Core", "members": ["S"]}))

    manual = asyncio.run(
        server_mod.call_tool(
            "set_module_view",
            {"model_id": "m1", "module_name": "Core", "x": 300, "y": 250, "width": 200, "height": 120},
        )
    )
    assert "Set module view for 'Core'" in _tool_text(manual)

    auto = asyncio.run(
        server_mod.call_tool(
            "auto_place_module_boxes",
            {"model_id": "m1", "only_missing": True},
        )
    )
    assert "Auto-placed module boxes" in _tool_text(auto)


def test_server_set_module_style_tool(monkeypatch):
    """Server should set module style and include it in module listing."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 1414)

    asyncio.run(server_mod.call_tool("create_model", {"name": "ModuleStyleServer", "model_id": "m1"}))
    asyncio.run(server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S", "initial_value": "100"}))
    asyncio.run(server_mod.call_tool("create_module", {"model_id": "m1", "name": "Core", "members": ["S"]}))

    styled = asyncio.run(
        server_mod.call_tool(
            "set_module_style",
            {
                "model_id": "m1",
                "module_name": "Core",
                "border_color": "#666666",
                "background": "#FFF7E6",
                "font_color": "#333333",
                "font_size": "10pt",
                "label_side": "top",
            },
        )
    )
    assert "Set module style for 'Core'" in _tool_text(styled)

    listed = asyncio.run(server_mod.call_tool("list_modules", {"model_id": "m1"}))
    assert "style=(border_color=#666666" in _tool_text(listed)


def test_module_lifecycle_methods():
    """Module lifecycle operations should update module state safely."""
    model = StellaModel("ModuleLifecycle")
    model.add_stock("S", "100")
    model.add_aux("k", "0.1")
    model.create_module("Core", members=["S", "k"])

    model.remove_from_module("Core", ["k"])
    assert model.modules["Core"].members == ["S"]

    model.rename_module("Core", "Renamed Core")
    assert "Core" not in model.modules
    assert "Renamed_Core" in model.modules

    deleted = model.delete_module("Renamed Core")
    assert deleted.name == "Renamed Core"
    assert not model.modules


def test_server_module_lifecycle_tools(monkeypatch):
    """Server module lifecycle tools should perform expected actions."""
    server_mod._session_models.clear()
    monkeypatch.setattr(server_mod, "_get_session_key", lambda: 1313)

    asyncio.run(server_mod.call_tool("create_model", {"name": "LifecycleServer", "model_id": "m1"}))
    asyncio.run(server_mod.call_tool("add_stock", {"model_id": "m1", "name": "S", "initial_value": "100"}))
    asyncio.run(server_mod.call_tool("add_aux", {"model_id": "m1", "name": "k", "equation": "0.1"}))
    asyncio.run(
        server_mod.call_tool(
            "create_module",
            {"model_id": "m1", "name": "Core", "members": ["S", "k"]},
        )
    )

    removed = asyncio.run(
        server_mod.call_tool(
            "remove_from_module",
            {"model_id": "m1", "module_name": "Core", "members": ["k"]},
        )
    )
    assert "total members: 1" in _tool_text(removed)

    renamed = asyncio.run(
        server_mod.call_tool(
            "rename_module",
            {"model_id": "m1", "module_name": "Core", "new_name": "Renamed Core"},
        )
    )
    assert "Renamed module 'Core' to 'Renamed Core'" in _tool_text(renamed)

    listed = asyncio.run(server_mod.call_tool("list_modules", {"model_id": "m1"}))
    assert "Renamed Core: S" in _tool_text(listed)

    deleted = asyncio.run(
        server_mod.call_tool(
            "delete_module",
            {"model_id": "m1", "module_name": "Renamed Core"},
        )
    )
    assert "Deleted module 'Renamed Core'" in _tool_text(deleted)


def test_validator_detects_empty_and_stale_modules():
    """Validator should flag empty modules and stale module members."""
    model = StellaModel("ModuleValidation")
    model.add_stock("S", "100")
    model.create_module("Empty", members=[])
    stale = model.create_module("Stale", members=["S"])
    stale.members.append("ghost_var")

    issues = validate_model(model)
    categories = {(issue.category, issue.severity) for issue in issues}
    assert ("module_empty", "warning") in categories
    assert ("module_member_missing", "error") in categories


def test_set_module_style_validation():
    """Model style setter should require updates and validate label side."""
    model = StellaModel("ModuleStyleValidation")
    model.add_stock("S", "100")
    model.create_module("Core", members=["S"])

    try:
        model.set_module_style("Core")
        raise AssertionError("Expected ValueError when no style fields are provided")
    except ValueError as exc:
        assert "At least one module style field" in str(exc)

    try:
        model.set_module_style("Core", label_side="diagonal")
        raise AssertionError("Expected ValueError for invalid label_side")
    except ValueError as exc:
        assert "label_side" in str(exc)
