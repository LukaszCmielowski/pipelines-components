"""Compile per-pattern documents indexing pipeline YAML from a base template."""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Any

import yaml

_PATTERN_CONFIG_PATH = Path(__file__).resolve().parent / "pattern_config.py"
_pattern_config_spec = importlib.util.spec_from_file_location("_autorag_pattern_config", _PATTERN_CONFIG_PATH)
if _pattern_config_spec is None or _pattern_config_spec.loader is None:
    raise ImportError(f"Cannot load pattern_config from {_PATTERN_CONFIG_PATH}")
_pattern_config_module = importlib.util.module_from_spec(_pattern_config_spec)
_pattern_config_spec.loader.exec_module(_pattern_config_module)
pattern_json_to_indexing_params = _pattern_config_module.pattern_json_to_indexing_params

RELATED_IMAGE_ENV_PREFIX = "RELATED_IMAGE_"
_BASE_TEMPLATE_NAME = "documents_indexing_pipeline.yaml"


def resolve_autorag_image(*, base_template_path: Path | None = None, fallback_image: str = "") -> str:
    """Return the AutoRAG runtime image from RELATED_IMAGE env or the base template."""
    for key, value in os.environ.items():
        if key.startswith(RELATED_IMAGE_ENV_PREFIX) and value.strip():
            return value.strip()

    if base_template_path is not None and base_template_path.is_file():
        image = _first_executor_image(base_template_path)
        if image:
            return image

    if fallback_image:
        return fallback_image

    raise ValueError(
        "Could not resolve AutoRAG image: set a RELATED_IMAGE_* environment variable "
        f"or provide a base template at {base_template_path}"
    )


def _first_executor_image(template_path: Path) -> str | None:
    """Extract the first container image from a compiled KFP YAML template."""
    content = template_path.read_text(encoding="utf-8")
    match = re.search(r"\n        image: (.+)\n", content)
    return match.group(1).strip() if match else None


def _coerce_default_value(value: Any, parameter_type: str) -> Any:
    if value is None:
        return None
    if parameter_type in ("NUMBER_INTEGER", "NUMBER_DOUBLE"):
        return float(value)
    if parameter_type == "BOOLEAN":
        return bool(value)
    if parameter_type == "STRUCT":
        return value if isinstance(value, dict) else {}
    return value


def _patch_parameter_defaults(parameters: dict[str, Any], defaults: dict[str, Any]) -> None:
    for name, value in defaults.items():
        if name not in parameters:
            continue
        param_def = parameters[name]
        if value is None:
            param_def.pop("defaultValue", None)
            continue
        param_type = param_def.get("parameterType", "STRING")
        param_def["defaultValue"] = _coerce_default_value(value, param_type)


def _patch_executor_images(deployment_spec: dict[str, Any], image: str) -> None:
    for executor in deployment_spec.get("executors", {}).values():
        container = executor.get("container")
        if isinstance(container, dict) and "image" in container:
            container["image"] = image


def _patch_pipeline_spec(spec: dict[str, Any], *, pattern_name: str, defaults: dict[str, Any], image: str) -> None:
    pipeline_info = spec.setdefault("pipelineInfo", {})
    if pattern_name:
        safe_name = re.sub(r"[^a-zA-Z0-9-]+", "-", pattern_name).strip("-").lower()
        pipeline_info["name"] = f"autorag-indexing-{safe_name}" if safe_name else pipeline_info.get("name")
        pipeline_info["displayName"] = f"AutoRAG Indexing — {pattern_name}"

    root_params = spec.get("root", {}).get("inputDefinitions", {}).get("parameters", {})
    _patch_parameter_defaults(root_params, defaults)

    deployment_spec = spec.get("deploymentSpec", {})
    if isinstance(deployment_spec, dict):
        _patch_executor_images(deployment_spec, image)


def _load_template_docs(template_path: Path) -> list[dict[str, Any]]:
    with template_path.open(encoding="utf-8") as handle:
        docs = [doc for doc in yaml.safe_load_all(handle) if isinstance(doc, dict)]
    if not docs:
        raise ValueError(f"Pipeline template {template_path} contains no YAML documents")
    return docs


def compile_indexing_pipeline_yaml(
    *,
    pattern: dict[str, Any],
    output_path: Path,
    base_template_path: Path,
    input_data_key: str = "",
    image: str | None = None,
) -> None:
    """Write a per-pattern indexing pipeline YAML with pattern defaults baked in.

    Args:
        pattern: Parsed pattern.json for one optimized RAG pattern.
        output_path: Destination path (e.g. ``<pattern_dir>/indexing_pipeline.yaml``).
        base_template_path: Canonical compiled ``documents_indexing_pipeline.yaml``.
        input_data_key: S3 prefix used during optimization (default for full corpus).
        image: AutoRAG container image digest; resolved from env/template when omitted.
    """
    defaults = pattern_json_to_indexing_params(pattern, input_data_key=input_data_key)
    pattern_name = defaults.get("pattern_name", pattern.get("name", ""))
    resolved_image = image or resolve_autorag_image(base_template_path=base_template_path)

    docs = _load_template_docs(base_template_path)
    pipeline_spec = docs[0]
    _patch_pipeline_spec(
        pipeline_spec,
        pattern_name=pattern_name,
        defaults=defaults,
        image=resolved_image,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.dump(pipeline_spec, handle, sort_keys=False)
        if len(docs) > 1:
            handle.write("---\n")
            yaml.dump(docs[1], handle, sort_keys=False)


def default_base_template_path(shared_root: Path) -> Path:
    """Return the embedded base template path under ``autorag.shared``."""
    return shared_root / "pipeline_templates" / _BASE_TEMPLATE_NAME
