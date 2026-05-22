"""Compatibility hardening tests for XMILE import/export edge cases."""

import pytest

from stella_mcp.validator import validate_model
from stella_mcp.xmile import Connector, StellaModel, parse_stmx


def test_dt_export_uses_safe_reciprocal_only_when_exact():
    """Non-exact reciprocal dt should be exported as plain dt value."""
    model = StellaModel("DtPrecision")
    model.sim_specs.dt = 0.3
    xml = model.to_xml(auto_layout=False)
    assert "<dt>0.3</dt>" in xml
    assert 'reciprocal="true"' not in xml

    model.sim_specs.dt = 0.25
    xml2 = model.to_xml(auto_layout=False)
    assert '<dt reciprocal="true">4</dt>' in xml2


def test_parse_normalizes_stock_flow_links_and_connectors(tmp_path):
    """Parser should normalize display-name references to internal identifiers."""
    xml = """<?xml version="1.0" encoding="utf-8"?>
<xmile version="1.0" xmlns="http://docs.oasis-open.org/xmile/ns/XMILE/v1.0" xmlns:isee="http://iseesystems.com/XMILE">
  <header><name>Compat</name></header>
  <sim_specs method="Euler" time_units="Years"><start>0</start><stop>10</stop><dt reciprocal="true">4</dt></sim_specs>
  <model>
    <variables>
      <stock name="Main Stock">
        <eqn>100</eqn>
        <outflow>total loss</outflow>
      </stock>
      <flow name="total loss">
        <eqn>1</eqn>
      </flow>
    </variables>
    <views>
      <view type="stock_flow">
        <connector uid="1" angle="0">
          <from>Main Stock</from>
          <to>total loss</to>
        </connector>
      </view>
    </views>
  </model>
</xmile>
"""
    path = tmp_path / "compat_links.stmx"
    path.write_text(xml, encoding="utf-8")

    model = parse_stmx(str(path))
    assert "Main_Stock" in model.stocks
    assert "total_loss" in model.flows
    assert model.stocks["Main_Stock"].outflows == ["total_loss"]
    assert model.flows["total_loss"].from_stock == "Main_Stock"
    assert model.connectors[0].from_var == "Main_Stock"
    assert model.connectors[0].to_var == "total_loss"


def test_parse_malformed_dt_reciprocal_keeps_default_dt(tmp_path):
    """Malformed reciprocal dt should not crash parsing."""
    xml = """<?xml version="1.0" encoding="utf-8"?>
<xmile version="1.0" xmlns="http://docs.oasis-open.org/xmile/ns/XMILE/v1.0">
  <header><name>BadDt</name></header>
  <sim_specs method="Euler" time_units="Years"><start>0</start><stop>10</stop><dt reciprocal="true">0</dt></sim_specs>
  <model><variables/></model>
</xmile>
"""
    path = tmp_path / "bad_dt.stmx"
    path.write_text(xml, encoding="utf-8")

    model = parse_stmx(str(path))
    assert model.sim_specs.dt == 0.25
    assert model.compatibility_warnings


def test_parse_strict_raises_on_compatibility_issue(tmp_path):
    """Strict parse mode should fail on malformed compatibility data."""
    xml = """<?xml version="1.0" encoding="utf-8"?>
<xmile version="1.0" xmlns="http://docs.oasis-open.org/xmile/ns/XMILE/v1.0">
  <header><name>BadDtStrict</name></header>
  <sim_specs method="Euler" time_units="Years"><start>0</start><stop>10</stop><dt reciprocal="true">0</dt></sim_specs>
  <model><variables/></model>
</xmile>
"""
    path = tmp_path / "bad_dt_strict.stmx"
    path.write_text(xml, encoding="utf-8")

    with pytest.raises(ValueError, match="sim_specs.dt reciprocal value must be > 0"):
        parse_stmx(str(path), compat_mode="strict")


def test_validator_flags_connectors_with_missing_endpoints():
    """Compatibility check should report connectors that target missing variables."""
    model = StellaModel("BrokenConnector")
    model.add_stock("S", "100")
    model.connectors.append(Connector(uid=1, from_var="ghost", to_var="S"))

    issues = validate_model(model)
    categories = {(issue.category, issue.severity) for issue in issues}
    assert ("connector_endpoint_missing", "error") in categories


def test_to_xml_permissive_and_strict_modes():
    """Export should warn in permissive mode and fail in strict mode for incompatibilities."""
    model = StellaModel("CompatExport")
    model.sim_specs.dt = -1
    model.add_stock("S", "100")
    model.connectors.append(Connector(uid=1, from_var="ghost", to_var="S"))

    xml = model.to_xml(auto_layout=False, compat_mode="permissive")
    assert "<dt reciprocal=\"true\">4</dt>" in xml
    assert model.last_export_warnings

    with pytest.raises(ValueError, match="sim_specs.dt=-1"):
        model.to_xml(auto_layout=False, compat_mode="strict")


def test_round_trip_preserves_unknown_attrs_and_elements(tmp_path):
    """Unknown XML attrs/elements should survive parse->to_xml round-trip."""
    xml = """<?xml version="1.0" encoding="utf-8"?>
<xmile version="1.0" xmlns="http://docs.oasis-open.org/xmile/ns/XMILE/v1.0" xmlns:isee="http://iseesystems.com/XMILE">
  <header>
    <smile version="1.0" namespace="std, isee"/>
    <name>PreserveUnknown</name>
    <uuid>1234</uuid>
    <vendor>v</vendor>
    <product version="1">p</product>
    <notes priority="high">keep me</notes>
  </header>
  <sim_specs method="Euler" time_units="Years" custom_attr="yes">
    <start>0</start><stop>10</stop><dt>1</dt>
    <custom_node foo="bar"/>
  </sim_specs>
  <isee:prefs keep="true" foo="bar"/>
  <model>
    <variables>
      <stock name="S" custom_stock_attr="1"><eqn>100</eqn><mystery/></stock>
      <flow name="f" custom_flow_attr="2"><eqn>1</eqn><flow_extra/></flow>
      <aux name="a" custom_aux_attr="3"><eqn>f</eqn><aux_extra/></aux>
      <group name="G" custom_group_attr="4"><entity name="S"/><group_extra/></group>
    </variables>
    <views>
      <style color="orange"><text_box color="green"/></style>
      <style color="purple"/>
      <view type="stock_flow" custom_view_attr="1">
        <style color="red"><stock color="black"/></style>
        <style color="blue"/>
        <group name="G" x="1" y="2" width="3" height="4" custom_group_view_attr="yes"><gv_extra/></group>
        <stock name="S" x="10" y="20" width="45" height="35" custom_stock_view_attr="yes"/>
        <flow name="f" x="15" y="20" custom_flow_view_attr="yes"><flow_view_extra/></flow>
        <aux name="a" x="30" y="40" custom_aux_view_attr="yes"/>
        <connector uid="1" angle="0" custom_conn_attr="yes"><from>S</from><to>f</to><conn_extra/></connector>
        <view_extra/>
      </view>
      <view type="table"/>
    </views>
    <model_extra/>
  </model>
</xmile>
"""
    path = tmp_path / "preserve_unknown.stmx"
    path.write_text(xml, encoding="utf-8")

    model = parse_stmx(str(path), compat_mode="permissive")
    exported = model.to_xml(auto_layout=False, compat_mode="permissive")

    for marker in [
        "keep me",
        "notes",
        "custom_attr=\"yes\"",
        "custom_node",
        "keep=\"true\"",
        "custom_stock_attr=\"1\"",
        "mystery",
        "custom_flow_attr=\"2\"",
        "flow_extra",
        "custom_aux_attr=\"3\"",
        "aux_extra",
        "custom_group_attr=\"4\"",
        "group_extra",
        "custom_view_attr=\"1\"",
        "custom_group_view_attr=\"yes\"",
        "gv_extra",
        "custom_stock_view_attr=\"yes\"",
        "custom_flow_view_attr=\"yes\"",
        "flow_view_extra",
        "custom_aux_view_attr=\"yes\"",
        "custom_conn_attr=\"yes\"",
        "conn_extra",
        "view_extra",
        "type=\"table\"",
        "model_extra",
    ]:
        assert marker in exported


def test_round_trip_preserves_unknown_namespaced_attrs(tmp_path):
    """Unknown namespaced attrs should be preserved with a declared prefix."""
    xml = """<?xml version="1.0" encoding="utf-8"?>
<xmile version="1.0" xmlns="http://docs.oasis-open.org/xmile/ns/XMILE/v1.0" xmlns:foo="http://example.com/ns">
  <header><name>Namespaced Extras</name></header>
  <sim_specs method="Euler" time_units="Years" foo:alpha="1"><start>0</start><stop>10</stop><dt>1</dt></sim_specs>
  <model>
    <variables>
      <stock name="S" foo:beta="2"><eqn>100</eqn></stock>
    </variables>
  </model>
</xmile>
"""
    path = tmp_path / "namespaced_attrs.stmx"
    path.write_text(xml, encoding="utf-8")

    model = parse_stmx(str(path), compat_mode="permissive")
    exported = model.to_xml(auto_layout=False, compat_mode="permissive")
    assert 'xmlns:ns1="http://example.com/ns"' in exported
    assert 'ns1:alpha="1"' in exported
    assert 'ns1:beta="2"' in exported


def test_module_view_group_with_only_extra_attrs_is_preserved():
    """Module view extras should serialize even with no geometry/style set."""
    model = StellaModel("ModuleViewExtras")
    model.add_stock("S", "100")
    model.create_module("M", members=["S"])
    module = model.modules["M"]
    module.view_extra_attrs = {"custom_flag": "yes"}
    module.view_extra_children_xml = ["<note>marker_module_note</note>"]

    xml = model.to_xml(auto_layout=False, compat_mode="permissive")
    assert 'custom_flag="yes"' in xml
    assert "marker_module_note" in xml
