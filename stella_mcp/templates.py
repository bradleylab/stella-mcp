"""Template management for Stella model starters."""

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

from .xmile import StellaModel, parse_stmx


_XMILE_NS = "http://docs.oasis-open.org/xmile/ns/XMILE/v1.0"
_TEMPLATE_NAME_CLEANER = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class TemplateInfo:
    """Metadata for a template entry."""

    name: str
    source: str  # "builtin" or "user"
    path: Path
    title: str
    description: str = ""
    tags: tuple[str, ...] = ()
    stocks: int = 0
    flows: int = 0
    auxiliaries: int = 0
    modules: int = 0
    updated_at: str = ""


def builtin_template_dir() -> Path:
    """Directory containing built-in shipped templates."""
    return Path(__file__).resolve().parent / "builtin_templates"


def user_template_dir() -> Path:
    """Directory containing user-defined templates."""
    configured = os.environ.get("STELLA_MCP_TEMPLATE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".stella-mcp" / "templates"


def _canonical_template_name(name: str) -> str:
    raw = name.strip()
    if not raw:
        raise ValueError("template_name cannot be empty")
    normalized = _TEMPLATE_NAME_CLEANER.sub("_", raw).strip("._-")
    if not normalized:
        raise ValueError("template_name must contain letters or numbers")
    return normalized.lower()


def _normalize_tag(tag: str) -> str:
    normalized = _TEMPLATE_NAME_CLEANER.sub("_", tag.strip().lower()).strip("._-")
    return normalized


def _normalize_tags(tags: Any) -> tuple[str, ...]:
    if tags is None:
        return ()
    if isinstance(tags, str):
        raw_tags = [tags]
    elif isinstance(tags, list):
        raw_tags = tags
    else:
        return ()

    normalized: list[str] = []
    for tag in raw_tags:
        cleaned = _normalize_tag(str(tag))
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized)


def _iter_template_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.glob("*.stmx") if path.is_file())


def _find_child(parent: ET.Element, tag: str) -> ET.Element | None:
    elem = parent.find(f"{{{_XMILE_NS}}}{tag}")
    if elem is None:
        elem = parent.find(tag)
    return elem


def _find_descendant(parent: ET.Element, tag: str) -> ET.Element | None:
    elem = parent.find(f".//{{{_XMILE_NS}}}{tag}")
    if elem is None:
        elem = parent.find(f".//{tag}")
    return elem


def _findall_children(parent: ET.Element, tag: str) -> list[ET.Element]:
    namespaced = parent.findall(f"{{{_XMILE_NS}}}{tag}")
    if namespaced:
        return namespaced
    return parent.findall(tag)


def _humanize_name(canonical_name: str) -> str:
    words = [part for part in canonical_name.replace("-", "_").split("_") if part]
    if not words:
        return canonical_name
    return " ".join(word.capitalize() for word in words)


def _metadata_sidecar_path(template_path: Path) -> Path:
    return template_path.with_suffix(".meta.json")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


@lru_cache(maxsize=1)
def _builtin_metadata() -> dict[str, dict[str, Any]]:
    """Read built-in template metadata manifest (if present)."""
    manifest_path = builtin_template_dir() / "metadata.json"
    parsed = _read_json(manifest_path)
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, meta in parsed.items():
        if not isinstance(meta, dict):
            continue
        try:
            canonical = _canonical_template_name(str(raw_name))
        except ValueError:
            continue
        normalized[canonical] = meta
    return normalized


def _template_shape(path: Path) -> dict[str, Any]:
    """Extract lightweight template stats directly from XMILE."""
    try:
        tree = ET.parse(path)
    except (OSError, ET.ParseError):
        return {
            "model_name": "",
            "stocks": 0,
            "flows": 0,
            "auxiliaries": 0,
            "modules": 0,
        }

    root = tree.getroot()
    model_name = ""
    header = _find_descendant(root, "header")
    if header is not None:
        name_elem = _find_child(header, "name")
        if name_elem is not None and name_elem.text:
            model_name = name_elem.text.strip()

    variables = _find_descendant(root, "variables")
    if variables is None:
        return {
            "model_name": model_name,
            "stocks": 0,
            "flows": 0,
            "auxiliaries": 0,
            "modules": 0,
        }

    return {
        "model_name": model_name,
        "stocks": len(_findall_children(variables, "stock")),
        "flows": len(_findall_children(variables, "flow")),
        "auxiliaries": len(_findall_children(variables, "aux")),
        "modules": len(_findall_children(variables, "group")),
    }


def _template_updated_at(path: Path) -> str:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return ""
    return modified.isoformat(timespec="seconds").replace("+00:00", "Z")


def _build_template_info(path: Path, source: str, metadata: dict[str, Any]) -> TemplateInfo:
    canonical_name = _canonical_template_name(path.stem)
    shape = _template_shape(path)
    title_raw = metadata.get("title")
    title = str(title_raw).strip() if isinstance(title_raw, str) else ""
    if not title:
        title = shape["model_name"] or _humanize_name(canonical_name)

    desc_raw = metadata.get("description")
    description = str(desc_raw).strip() if isinstance(desc_raw, str) else ""
    tags = _normalize_tags(metadata.get("tags"))

    return TemplateInfo(
        name=canonical_name,
        source=source,
        path=path,
        title=title,
        description=description,
        tags=tags,
        stocks=shape["stocks"],
        flows=shape["flows"],
        auxiliaries=shape["auxiliaries"],
        modules=shape["modules"],
        updated_at=_template_updated_at(path),
    )


def _collect_templates(directory: Path, source: str) -> dict[str, TemplateInfo]:
    templates: dict[str, TemplateInfo] = {}
    builtin_meta = _builtin_metadata() if source == "builtin" else {}
    for path in _iter_template_files(directory):
        canonical_name = _canonical_template_name(path.stem)
        metadata = builtin_meta.get(canonical_name, {})
        if source == "user":
            metadata = _read_json(_metadata_sidecar_path(path))
        templates[canonical_name] = _build_template_info(path, source=source, metadata=metadata)
    return templates


def list_templates(
    source: str | None = None,
    query: str | None = None,
    tags: list[str] | None = None,
) -> list[TemplateInfo]:
    """List available templates with optional source/query/tag filtering."""
    source_filter = source.lower().strip() if source else None
    if source_filter is not None and source_filter not in {"builtin", "user"}:
        raise ValueError("source must be one of: builtin, user")

    combined: dict[str, TemplateInfo] = {}
    if source_filter in (None, "builtin"):
        combined = _collect_templates(builtin_template_dir(), "builtin")
    if source_filter in (None, "user"):
        user_templates = _collect_templates(user_template_dir(), "user")
        if source_filter is None:
            combined.update(user_templates)
        else:
            combined = user_templates

    filtered = [combined[name] for name in sorted(combined)]

    normalized_query = (query or "").strip().lower()
    if normalized_query:
        filtered = [
            info
            for info in filtered
            if normalized_query in info.name.lower()
            or normalized_query in info.title.lower()
            or normalized_query in info.description.lower()
        ]

    required_tags = set(_normalize_tags(tags))
    if required_tags:
        filtered = [info for info in filtered if required_tags.issubset(set(info.tags))]

    return filtered


def resolve_template(template_name: str) -> TemplateInfo:
    """Find a template by name."""
    target = _canonical_template_name(template_name)
    by_name = {info.name: info for info in list_templates()}
    if target not in by_name:
        raise FileNotFoundError(f"Template '{template_name}' not found")
    return by_name[target]


def get_template_info(template_name: str) -> TemplateInfo:
    """Get rich metadata for a template by name."""
    return resolve_template(template_name)


def load_template_model(template_name: str) -> tuple[TemplateInfo, StellaModel]:
    """Load a template by name into a StellaModel."""
    info = resolve_template(template_name)
    return info, parse_stmx(str(info.path))


def save_user_template(
    template_name: str,
    model: StellaModel,
    overwrite: bool = False,
    description: str = "",
    tags: list[str] | None = None,
) -> TemplateInfo:
    """Save a model as a user-defined template."""
    canonical_name = _canonical_template_name(template_name)
    directory = user_template_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target_path = directory / f"{canonical_name}.stmx"

    if target_path.exists() and not overwrite:
        raise ValueError(f"Template '{canonical_name}' already exists. Set overwrite=true to replace it.")

    # Save template exactly as authored; avoid forcing a relayout.
    target_path.write_text(model.to_xml(auto_layout=False), encoding="utf-8")

    metadata_payload: dict[str, Any] = {"title": model.name}
    clean_description = description.strip()
    if clean_description:
        metadata_payload["description"] = clean_description
    clean_tags = list(_normalize_tags(tags))
    if clean_tags:
        metadata_payload["tags"] = clean_tags
    metadata_path = _metadata_sidecar_path(target_path)
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return get_template_info(canonical_name)
