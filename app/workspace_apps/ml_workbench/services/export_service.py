"""Export formatting and workbook generation for the ML Workbench app.

This module consumes the canonical structured export payload from
``modeling_service`` and derives workbook-friendly tabular views from it.
It intentionally avoids duplicating modeling or comparison logic.
"""

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from typing import Any

import pandas as pd

from app.workspace_apps.ml_workbench.services.modeling_service import build_export_payload


def get_canonical_export_payload() -> dict[str, Any]:
    """Return the canonical export payload used by every export target."""
    return build_export_payload()


def _json_cell(value: object) -> str:
    """Return a stable JSON string for nested workbook cell values."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _stringify_cell(value: object) -> str | int | float | bool | None:
    """Return an Excel-friendly scalar representation."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return _json_cell(value)


def _flatten_mapping_rows(
    value: object,
    *,
    section: str,
    prefix: str = "",
) -> list[dict[str, Any]]:
    """Flatten nested dict/list content into section/key/value rows."""
    rows: list[dict[str, Any]] = []

    if isinstance(value, dict):
        for key, child_value in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_mapping_rows(child_value, section=section, prefix=child_prefix))
        return rows

    if isinstance(value, list):
        if not value:
            rows.append({"section": section, "key": prefix, "value": "[]"})
            return rows
        for index, child_value in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            rows.extend(_flatten_mapping_rows(child_value, section=section, prefix=child_prefix))
        return rows

    rows.append({"section": section, "key": prefix, "value": _stringify_cell(value)})
    return rows


def build_summary_sheet_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return key/value rows for the workbook summary sheet."""
    bundle = dict(payload.get("results_bundle", {}))
    workspace = dict(bundle.get("workspace", {}))
    rows: list[dict[str, Any]] = [
        {"section": "export", "key": "exported_at", "value": payload.get("exported_at")},
        {"section": "export", "key": "export_version", "value": payload.get("export_version")},
        {"section": "workspace", "key": "problem_type", "value": workspace.get("problem_type")},
        {"section": "workspace", "key": "target_column", "value": workspace.get("target_column")},
        {"section": "workspace", "key": "source_dataset_name", "value": workspace.get("source_dataset_name")},
        {
            "section": "workspace",
            "key": "identifier_columns",
            "value": _json_cell(list(workspace.get("identifier_columns", []))),
        },
        {
            "section": "workspace",
            "key": "ignored_columns",
            "value": _json_cell(list(workspace.get("ignored_columns", []))),
        },
        {
            "section": "comparison",
            "key": "comparison_metric_name",
            "value": bundle.get("comparison_metric_name"),
        },
        {"section": "comparison", "key": "best_candidate_id", "value": bundle.get("best_candidate_id")},
        {"section": "comparison", "key": "active_candidate_id", "value": bundle.get("active_candidate_id")},
        {
            "section": "comparison",
            "key": "candidate_count",
            "value": len(list(bundle.get("candidates", []))),
        },
    ]
    return rows


def build_candidate_comparison_sheet_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one comparison-oriented row per candidate."""
    bundle = dict(payload.get("results_bundle", {}))
    rows: list[dict[str, Any]] = []
    for candidate in list(bundle.get("candidates", [])):
        run_record = candidate.get("run_record") if isinstance(candidate, dict) else None
        metrics = dict(run_record.get("metrics", {})) if isinstance(run_record, dict) else {}
        rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "candidate_label": candidate.get("candidate_label"),
                "model_id": candidate.get("model_id"),
                "status": run_record.get("status") if isinstance(run_record, dict) else None,
                "comparison_metric_name": candidate.get("comparison_metric_name"),
                "comparison_metric_value": candidate.get("comparison_metric_value"),
                "is_comparison_eligible": candidate.get("is_comparison_eligible"),
                "is_best_candidate": candidate.get("is_best_candidate"),
                "run_id": run_record.get("run_id") if isinstance(run_record, dict) else None,
                "training_mode": run_record.get("training_mode") if isinstance(run_record, dict) else None,
                "metric_count": len(metrics),
                "metrics_json": _json_cell(metrics),
            }
        )
    return rows


def build_candidate_settings_sheet_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return flattened candidate settings rows."""
    bundle = dict(payload.get("results_bundle", {}))
    rows: list[dict[str, Any]] = []
    for candidate in list(bundle.get("candidates", [])):
        config = dict(candidate.get("config", {})) if isinstance(candidate.get("config"), dict) else {}
        dataset_plan = (
            dict(candidate.get("dataset_plan", {})) if isinstance(candidate.get("dataset_plan"), dict) else {}
        )
        candidate_preprocessing = dict(dataset_plan.get("candidate_preprocessing", {}))
        rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "candidate_label": candidate.get("candidate_label"),
                "model_id": candidate.get("model_id"),
                "enabled": candidate.get("enabled"),
                "is_best_candidate": candidate.get("is_best_candidate"),
                "comparison_metric_name": candidate.get("comparison_metric_name"),
                "comparison_metric_value": candidate.get("comparison_metric_value"),
                "classification_threshold": config.get("classification_threshold"),
                "train_test_split_enabled": config.get("train_test_split_enabled"),
                "notes": config.get("notes"),
                "hyperparameters_json": _json_cell(
                    config.get("hyperparameters", config.get("custom_params", {}))
                ),
                "candidate_preprocessing_json": _json_cell(candidate_preprocessing),
                "dataset_plan_json": _json_cell(dataset_plan),
                "config_json": _json_cell(config),
            }
        )
    return rows


def build_predictor_columns_sheet_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one row per resolved predictor column per candidate."""
    bundle = dict(payload.get("results_bundle", {}))
    rows: list[dict[str, Any]] = []
    for candidate in list(bundle.get("candidates", [])):
        dataset_plan = (
            dict(candidate.get("dataset_plan", {})) if isinstance(candidate.get("dataset_plan"), dict) else {}
        )
        run_record = candidate.get("run_record") if isinstance(candidate, dict) else None
        predictor_columns = list(dataset_plan.get("resolved_feature_columns", []))
        if not predictor_columns and isinstance(run_record, dict):
            predictor_columns = list(run_record.get("feature_columns", []))

        if not predictor_columns:
            rows.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate_label": candidate.get("candidate_label"),
                    "model_id": candidate.get("model_id"),
                    "predictor_column": None,
                    "source": None,
                }
            )
            continue

        for column_name in predictor_columns:
            rows.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate_label": candidate.get("candidate_label"),
                    "model_id": candidate.get("model_id"),
                    "predictor_column": column_name,
                    "source": "dataset_plan" if dataset_plan.get("resolved_feature_columns") else "run_record",
                }
            )
    return rows


def build_shared_preprocessing_sheet_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return flattened rows for workspace-level preprocessing settings."""
    bundle = dict(payload.get("results_bundle", {}))
    return _flatten_mapping_rows(
        dict(bundle.get("shared_preprocessing", {})),
        section="shared_preprocessing",
    )


def build_feature_engineering_sheet_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one row per feature specification when available."""
    bundle = dict(payload.get("results_bundle", {}))
    feature_engineering = dict(bundle.get("feature_engineering", {}))
    feature_specs = list(feature_engineering.get("feature_specs", []))
    if not feature_specs:
        return [
            {
                "feature_index": None,
                "feature_id": None,
                "feature_name": None,
                "feature_type": None,
                "feature_spec_json": "[]",
            }
        ]

    rows: list[dict[str, Any]] = []
    for index, feature_spec in enumerate(feature_specs, start=1):
        feature_spec_dict = dict(feature_spec) if isinstance(feature_spec, dict) else {"value": feature_spec}
        rows.append(
            {
                "feature_index": index,
                "feature_id": feature_spec_dict.get("feature_id"),
                "feature_name": feature_spec_dict.get("feature_name", feature_spec_dict.get("name")),
                "feature_type": feature_spec_dict.get("feature_type", feature_spec_dict.get("kind")),
                "feature_spec_json": _json_cell(feature_spec_dict),
            }
        )
    return rows


def build_raw_json_sheet_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one row per line of the canonical JSON payload."""
    serialized_payload = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    return [
        {"line_number": line_number, "json_line": line}
        for line_number, line in enumerate(serialized_payload.splitlines(), start=1)
    ]


def build_excel_sheet_rows(payload: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Return all workbook sheet rows derived from the canonical payload."""
    resolved_payload = payload if payload is not None else get_canonical_export_payload()
    return {
        "Summary": build_summary_sheet_rows(resolved_payload),
        "Candidate Comparison": build_candidate_comparison_sheet_rows(resolved_payload),
        "Candidate Settings": build_candidate_settings_sheet_rows(resolved_payload),
        "Predictor Columns": build_predictor_columns_sheet_rows(resolved_payload),
        "Shared Preprocessing": build_shared_preprocessing_sheet_rows(resolved_payload),
        "Feature Engineering": build_feature_engineering_sheet_rows(resolved_payload),
        "Raw JSON": build_raw_json_sheet_rows(resolved_payload),
    }


def _sheet_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a stable dataframe for one workbook sheet."""
    if not rows:
        return pd.DataFrame([{}])
    return pd.DataFrame(rows)


def _default_export_filename(payload: dict[str, Any]) -> str:
    """Return a timestamped default filename for workbook exports."""
    exported_at = str(payload.get("exported_at") or "").strip()
    if exported_at:
        timestamp = exported_at.replace(":", "-")
    else:
        timestamp = datetime.utcnow().isoformat(timespec="seconds").replace(":", "-")
    return f"ml_workbench_export_{timestamp}.xlsx"


def generate_excel_export_workbook(
    payload: dict[str, Any] | None = None,
    *,
    filename: str | None = None,
) -> tuple[bytes, str]:
    """Return workbook bytes plus a suggested filename for future download use."""
    resolved_payload = payload if payload is not None else get_canonical_export_payload()
    workbook_buffer = BytesIO()
    sheet_rows = build_excel_sheet_rows(resolved_payload)

    with pd.ExcelWriter(workbook_buffer, engine="openpyxl") as writer:
        for sheet_name, rows in sheet_rows.items():
            _sheet_dataframe(rows).to_excel(writer, sheet_name=sheet_name, index=False)

    return workbook_buffer.getvalue(), (filename or _default_export_filename(resolved_payload))
