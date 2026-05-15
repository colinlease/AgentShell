from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.workspace_apps.ml_workbench.constants import CREATED_BY_AGENT, STAGE_RESULTS, SUPPORTED_PROBLEM_TYPES
from app.workspace_apps.ml_workbench.schemas import CandidateModelConfig, FeatureSpec
from app.workspace_apps.ml_workbench.services import dataset_service, feature_service, modeling_service, preprocessing_service
from app.workspace_apps.ml_workbench.state import (
    append_feature_spec,
    get_active_candidate_id,
    get_app_state,
    get_best_candidate_id,
    get_candidate_models,
    get_feature_spec,
    get_feature_specs,
    get_preprocessing_config,
    get_status_flags,
    get_state_value,
    update_feature_spec,
    update_preprocessing_config,
    update_state_values,
)


class MLWorkbenchToolServiceError(RuntimeError):
    """Raised when an ML Workbench agent tool request cannot be completed."""


class MLWorkbenchToolService:
    """Thin agent-facing orchestration layer for ML Workbench tools."""

    def _error(self, message: str, *, errors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "error",
            "message": message,
        }
        if errors:
            payload["errors"] = errors
        return payload

    def _normalize_column_list(self, value: object) -> list[str]:
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            raise MLWorkbenchToolServiceError("Expected a list of column names.")
        normalized: list[str] = []
        for item in value:
            column_name = str(item).strip()
            if column_name and column_name not in normalized:
                normalized.append(column_name)
        return normalized

    def _require_loaded_columns(self) -> list[str]:
        active_dataset_name = dataset_service.get_active_dataset_name(default_to_working=True)
        if not active_dataset_name:
            raise MLWorkbenchToolServiceError("No active dataset is available in ML Workbench.")
        summary = dataset_service.dataset_summary(active_dataset_name)
        column_names = list(summary.get("column_names", []))
        if not column_names:
            raise MLWorkbenchToolServiceError("The active dataset does not expose any columns.")
        return column_names

    def _validate_columns_exist(
        self,
        *,
        columns: list[str],
        valid_columns: list[str],
        field_name: str,
    ) -> None:
        invalid_columns = [column for column in columns if column not in valid_columns]
        if invalid_columns:
            raise MLWorkbenchToolServiceError(
                f"{field_name} contains unknown columns: {', '.join(invalid_columns)}."
            )

    def _working_data_summary(self) -> dict[str, Any] | None:
        dataset_name = dataset_service.get_active_dataset_name(default_to_working=True)
        if not dataset_name:
            return None
        try:
            summary = dataset_service.dataset_summary(dataset_name)
        except Exception:
            return None
        return {
            "artifact_name": str(summary.get("name", dataset_name)),
            "rows": int(summary.get("rows", 0)),
            "columns": int(summary.get("columns", 0)),
            "ready_for_modeling": bool(summary.get("ready_for_modeling", False)),
        }

    def _rebuild_working_data_summary(self) -> tuple[dict[str, Any] | None, list[str]]:
        execution_result = preprocessing_service.rebuild_working_data_from_shared_rules()
        return self._working_data_summary(), list(execution_result.warnings)

    def _compact_feature_spec(self, feature_spec: FeatureSpec) -> dict[str, Any]:
        validation = feature_spec.get("validation", {})
        warnings = list(validation.get("warnings", [])) if isinstance(validation, dict) else []
        return {
            "feature_id": str(feature_spec.get("feature_id", "")),
            "feature_name": str(feature_spec.get("feature_name", "")),
            "feature_type": feature_spec.get("feature_type"),
            "operation_family": feature_spec.get("operation_family"),
            "operation": feature_spec.get("operation"),
            "source_columns": list(feature_spec.get("source_columns", [])),
            "enabled": bool(feature_spec.get("enabled", True)),
            "status": str(feature_spec.get("status", "")),
            "validation": {
                "is_valid": bool(validation.get("is_valid", not warnings)) if isinstance(validation, dict) else True,
                "warnings": warnings,
            },
        }

    def _primary_metric_summary(self, run_record: dict[str, Any] | None) -> tuple[str | None, float | None]:
        if not isinstance(run_record, dict):
            return None, None
        metrics = run_record.get("metrics", {})
        if not isinstance(metrics, dict) or not metrics:
            return None, None
        preferred_names = ["roc_auc", "f1", "accuracy", "precision", "recall", "rmse", "mae", "r2"]
        for metric_name in preferred_names:
            metric_value = metrics.get(metric_name)
            if isinstance(metric_value, (int, float)):
                return metric_name, float(metric_value)
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                return str(metric_name), float(metric_value)
        return None, None

    def _compact_candidate(self, candidate: CandidateModelConfig) -> dict[str, Any]:
        run_record = candidate.get("latest_run_record")
        primary_metric_name, primary_metric_value = self._primary_metric_summary(run_record)
        return {
            "candidate_id": str(candidate.get("candidate_id", "")),
            "candidate_label": str(candidate.get("candidate_label", "")),
            "model_id": str(candidate.get("model_id", "")),
            "enabled": bool(candidate.get("enabled", False)),
            "train_test_split_enabled": bool(candidate.get("train_test_split_enabled", False)),
            "latest_run_id": candidate.get("latest_run_id"),
            "latest_run_status": str(run_record.get("status", "not_run")) if isinstance(run_record, dict) else "not_run",
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": primary_metric_value,
        }

    def _normalize_model_param(self, key: str, value: object, schema: dict[str, Any]) -> object:
        field_type = str(schema.get("type", "")).strip().lower()
        options = schema.get("options")
        if value is None:
            if field_type.startswith("optional"):
                return None
            return None
        if field_type in {"float"}:
            if isinstance(value, bool):
                raise MLWorkbenchToolServiceError(f"Parameter '{key}' must be a float.")
            return float(value)
        if field_type in {"int", "optional_int"}:
            if isinstance(value, bool):
                raise MLWorkbenchToolServiceError(f"Parameter '{key}' must be an integer.")
            return int(value)
        if field_type == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "yes", "1"}:
                    return True
                if lowered in {"false", "no", "0"}:
                    return False
            raise MLWorkbenchToolServiceError(f"Parameter '{key}' must be a boolean.")
        if field_type in {"select", "str"}:
            normalized = str(value)
            if isinstance(options, list) and normalized not in options:
                raise MLWorkbenchToolServiceError(
                    f"Parameter '{key}' must be one of: {', '.join(str(option) for option in options)}."
                )
            return normalized
        return value

    def _validate_custom_params(self, *, model_id: str, custom_params: object) -> dict[str, Any]:
        if custom_params in (None, ""):
            return {}
        if not isinstance(custom_params, dict):
            raise MLWorkbenchToolServiceError("custom_params must be an object.")
        specs = modeling_service.list_available_model_specs()
        spec = next((item for item in specs if str(item.get("model_id", "")) == model_id), None)
        if spec is None:
            raise MLWorkbenchToolServiceError(f"Model '{model_id}' is not registered.")
        param_schema = spec.get("param_schema", {})
        if not isinstance(param_schema, dict):
            return dict(custom_params)
        normalized: dict[str, Any] = {}
        unknown_keys = [key for key in custom_params.keys() if key not in param_schema]
        if unknown_keys:
            allowed = sorted(param_schema.keys())
            raise MLWorkbenchToolServiceError(
                f"Invalid model parameters: {', '.join(unknown_keys)}. Allowed parameters: {', '.join(allowed)}."
            )
        for key, value in custom_params.items():
            schema = param_schema.get(key, {})
            if isinstance(schema, dict):
                normalized[key] = self._normalize_model_param(str(key), value, schema)
            else:
                normalized[key] = value
        return normalized

    def _normalize_candidate_strategy_value(self, field_name: str, value: object) -> str | None:
        """Normalize one candidate-preprocessing strategy into canonical text."""
        if value in (None, ""):
            return None
        if isinstance(value, dict):
            value = value.get("strategy")
            if value in (None, ""):
                return None

        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if not normalized:
            return None

        if field_name == "encoding":
            if normalized in {"onehot", "one_hot"}:
                return "one_hot"
            if normalized == "none":
                return "none"
            return normalized

        if field_name == "scaling":
            if normalized in {"standard", "standard_scaler", "zscore", "z_score"}:
                return "standard"
            if normalized in {"minmax", "min_max"}:
                return "minmax"
            if normalized == "none":
                return "none"
            return normalized

        if field_name == "class_rebalancing":
            return normalized

        return normalized

    def _normalize_candidate_preprocessing_update(
        self,
        preprocessing: object,
    ) -> tuple[dict[str, Any], list[str]]:
        """Normalize agent-written candidate preprocessing into canonical keys."""
        if not isinstance(preprocessing, dict):
            raise MLWorkbenchToolServiceError("preprocessing must be an object.")

        normalized_preprocessing = deepcopy(preprocessing)
        warnings: list[str] = []

        for legacy_field, canonical_field in (
            ("encoding", "encoding_strategy"),
            ("scaling", "scaling_strategy"),
            ("class_rebalancing", "class_rebalancing_strategy"),
        ):
            canonical_value = self._normalize_candidate_strategy_value(
                legacy_field,
                normalized_preprocessing.get(canonical_field),
            )
            legacy_value = self._normalize_candidate_strategy_value(
                legacy_field,
                normalized_preprocessing.get(legacy_field),
            )

            if canonical_value is not None:
                normalized_preprocessing[canonical_field] = canonical_value
                if legacy_value is not None and legacy_value != canonical_value:
                    warnings.append(
                        f"Ignored preprocessing.{legacy_field} because preprocessing.{canonical_field} was already set."
                    )
            elif legacy_value is not None:
                normalized_preprocessing[canonical_field] = legacy_value
                warnings.append(
                    f"Normalized preprocessing.{legacy_field} to preprocessing.{canonical_field}."
                )

            if legacy_field in normalized_preprocessing and not isinstance(
                normalized_preprocessing.get(legacy_field),
                dict,
            ):
                normalized_preprocessing.pop(legacy_field, None)

        return normalized_preprocessing, warnings

    def get_modeling_setup(self) -> dict[str, Any]:
        state = get_app_state()
        return {
            "status": "ok",
            "setup": {
                "active_dataset_name": state.get("active_dataset_name"),
                "problem_type": state.get("problem_type"),
                "target_column": state.get("target_column"),
                "positive_class_label": state.get("positive_class_label"),
                "id_columns": list(state.get("id_columns", [])),
                "ignored_columns": list(state.get("ignored_columns", [])),
                "selected_feature_columns": list(state.get("selected_feature_columns", [])),
            },
            "status_flags": deepcopy(get_status_flags()),
            "warnings": [],
        }

    def set_modeling_setup(self, **updates: object) -> dict[str, Any]:
        valid_columns = self._require_loaded_columns()
        state = get_app_state()
        normalized_updates: dict[str, Any] = {}

        if "problem_type" in updates and updates.get("problem_type") is not None:
            problem_type = str(updates.get("problem_type")).strip().lower()
            if problem_type not in SUPPORTED_PROBLEM_TYPES:
                return self._error(
                    "Invalid problem type.",
                    errors=[{"field": "problem_type", "reason": f"Expected one of {SUPPORTED_PROBLEM_TYPES}."}],
                )
            normalized_updates["problem_type"] = problem_type

        if "target_column" in updates:
            target_column = updates.get("target_column")
            if target_column in (None, ""):
                normalized_updates["target_column"] = None
            else:
                normalized_target = str(target_column).strip()
                self._validate_columns_exist(
                    columns=[normalized_target],
                    valid_columns=valid_columns,
                    field_name="target_column",
                )
                normalized_updates["target_column"] = normalized_target

        if "positive_class_label" in updates:
            normalized_updates["positive_class_label"] = updates.get("positive_class_label")

        if "id_columns" in updates and updates.get("id_columns") is not None:
            id_columns = self._normalize_column_list(updates.get("id_columns"))
            self._validate_columns_exist(columns=id_columns, valid_columns=valid_columns, field_name="id_columns")
            normalized_updates["id_columns"] = id_columns

        if "ignored_columns" in updates and updates.get("ignored_columns") is not None:
            ignored_columns = self._normalize_column_list(updates.get("ignored_columns"))
            self._validate_columns_exist(
                columns=ignored_columns,
                valid_columns=valid_columns,
                field_name="ignored_columns",
            )
            normalized_updates["ignored_columns"] = ignored_columns

        if "selected_feature_columns" in updates and updates.get("selected_feature_columns") is not None:
            selected_feature_columns = self._normalize_column_list(updates.get("selected_feature_columns"))
            self._validate_columns_exist(
                columns=selected_feature_columns,
                valid_columns=valid_columns,
                field_name="selected_feature_columns",
            )
            normalized_updates["selected_feature_columns"] = selected_feature_columns

        if normalized_updates.get("problem_type") != "classification" and "positive_class_label" in normalized_updates:
            if normalized_updates.get("problem_type") or state.get("problem_type") != "classification":
                normalized_updates["positive_class_label"] = None

        update_state_values(**normalized_updates)
        return {
            "status": "ok",
            "message": "Updated modeling setup.",
            "changed": normalized_updates,
            "setup": self.get_modeling_setup()["setup"],
            "warnings": [],
        }

    def get_preprocessing_config_summary(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "preprocessing_config": deepcopy(get_preprocessing_config()),
            "working_data": self._working_data_summary(),
            "warnings": [],
        }

    def update_preprocessing_config_summary(self, **updates: object) -> dict[str, Any]:
        changed: dict[str, Any] = {}
        if "drop_columns" in updates and updates.get("drop_columns") is not None:
            valid_columns = self._require_loaded_columns()
            drop_columns = self._normalize_column_list(updates.get("drop_columns"))
            self._validate_columns_exist(columns=drop_columns, valid_columns=valid_columns, field_name="drop_columns")
            changed["drop_columns"] = drop_columns

        for key in ("numeric_imputation", "categorical_imputation", "datetime_handling"):
            value = updates.get(key)
            if value is None:
                continue
            if not isinstance(value, dict):
                raise MLWorkbenchToolServiceError(f"{key} must be an object.")
            normalized_block = deepcopy(value)
            if "columns" in normalized_block and normalized_block.get("columns") is not None:
                valid_columns = self._require_loaded_columns()
                normalized_block["columns"] = self._normalize_column_list(normalized_block.get("columns"))
                self._validate_columns_exist(
                    columns=normalized_block["columns"],
                    valid_columns=valid_columns,
                    field_name=f"{key}.columns",
                )
            if "expanded_columns" in normalized_block and normalized_block.get("expanded_columns") is not None:
                valid_columns = self._require_loaded_columns()
                normalized_block["expanded_columns"] = self._normalize_column_list(normalized_block.get("expanded_columns"))
                self._validate_columns_exist(
                    columns=normalized_block["expanded_columns"],
                    valid_columns=valid_columns,
                    field_name=f"{key}.expanded_columns",
                )
            changed[key] = normalized_block

        if changed:
            update_preprocessing_config(**changed)

        rebuild = bool(updates.get("rebuild_working_data"))
        result_message = "Updated preprocessing configuration."
        warnings: list[str] = []
        working_data = self._working_data_summary()
        if rebuild:
            execution_result = preprocessing_service.rebuild_working_data_from_shared_rules()
            result_message = "Updated preprocessing configuration and rebuilt working data."
            warnings = list(execution_result.warnings)
            working_data = self._working_data_summary()

        return {
            "status": "ok",
            "message": result_message,
            "changed": changed,
            "working_data": working_data,
            "warnings": warnings,
        }

    def get_feature_specs_summary(self) -> dict[str, Any]:
        feature_specs = [self._compact_feature_spec(feature) for feature in get_feature_specs()]
        return {
            "status": "ok",
            "feature_specs": feature_specs,
            "feature_count": len(feature_specs),
            "warnings": [],
        }

    def upsert_feature_spec(self, **updates: object) -> dict[str, Any]:
        preview_only = bool(updates.get("preview_only"))
        feature_id = str(updates.get("feature_id") or "").strip() or None

        if feature_id:
            current_feature = get_feature_spec(feature_id)
            if current_feature is None:
                return self._error(
                    "Feature specification was not found.",
                    errors=[{"field": "feature_id", "reason": f"Unknown feature_id '{feature_id}'."}],
                )
            feature_spec = deepcopy(current_feature)
            for key, value in updates.items():
                if key in {"feature_id", "preview_only"} or value is None:
                    continue
                feature_spec[key] = value
        else:
            required_fields = ["feature_name", "feature_type", "operation_family", "operation"]
            missing_fields = [field for field in required_fields if not str(updates.get(field, "")).strip()]
            if missing_fields:
                return self._error(
                    "Missing required fields for feature creation.",
                    errors=[{"field": field, "reason": "This field is required."} for field in missing_fields],
                )
            feature_spec = feature_service.build_default_feature_spec(
                feature_name=str(updates.get("feature_name")),
                feature_type=str(updates.get("feature_type")),
                operation_family=str(updates.get("operation_family")),
                operation=str(updates.get("operation")),
                source_columns=self._normalize_column_list(updates.get("source_columns")),
                expression=str(updates.get("expression", "") or ""),
                parameters=deepcopy(updates.get("parameters")) if isinstance(updates.get("parameters"), dict) else None,
                builder_mode=str(updates.get("builder_mode", "guided") or "guided"),
                expression_language=str(updates.get("expression_language")) if updates.get("expression_language") else None,
                created_by=CREATED_BY_AGENT,
                description=str(updates.get("description")) if updates.get("description") is not None else None,
            )
            if updates.get("enabled") is not None:
                feature_spec["enabled"] = bool(updates.get("enabled"))
            if updates.get("apply_order") is not None:
                feature_spec["apply_order"] = int(updates.get("apply_order"))

        source_df = None
        try:
            dataset_name = dataset_service.get_active_dataset_name(default_to_working=True)
            if dataset_name:
                source_df = dataset_service.get_dataset_copy(dataset_name)
        except Exception:
            source_df = None
        preview_result = feature_service.preview_feature_spec(feature_spec, source_df=source_df)
        feature_spec["validation"] = deepcopy(preview_result.validation)
        feature_spec["status"] = "ready" if preview_result.validation.get("is_valid") else "invalid"

        if preview_only:
            return {
                "status": "ok",
                "message": "Previewed feature specification.",
                "feature_spec": self._compact_feature_spec(feature_spec),
                "validation": deepcopy(preview_result.validation),
                "warnings": list(preview_result.warnings),
            }

        if feature_id:
            updated_feature = update_feature_spec(feature_id, **feature_spec)
            if updated_feature is None:
                raise MLWorkbenchToolServiceError(f"Feature '{feature_id}' was not found.")
            stored_feature = updated_feature
            message = "Updated feature specification and rebuilt working data."
        else:
            append_feature_spec(feature_spec)
            stored_feature = feature_spec
            message = "Created feature specification and rebuilt working data."

        working_data, rebuild_warnings = self._rebuild_working_data_summary()
        warnings = list(preview_result.warnings) + rebuild_warnings

        return {
            "status": "ok",
            "message": message,
            "feature_spec": self._compact_feature_spec(stored_feature),
            "validation": deepcopy(preview_result.validation),
            "working_data": working_data,
            "warnings": warnings,
        }

    def remove_feature_spec_summary(self, feature_id: str) -> dict[str, Any]:
        removed = feature_service.remove_stored_feature_specs([feature_id])
        if not removed:
            return self._error(
                "Feature specification was not found.",
                errors=[{"field": "feature_id", "reason": f"Unknown feature_id '{feature_id}'."}],
            )
        working_data, rebuild_warnings = self._rebuild_working_data_summary()
        return {
            "status": "ok",
            "message": "Removed feature specification and rebuilt working data.",
            "removed_feature_id": feature_id,
            "remaining_feature_count": len(get_feature_specs()),
            "working_data": working_data,
            "warnings": rebuild_warnings,
        }

    def get_candidate_models_summary(self) -> dict[str, Any]:
        candidates = [self._compact_candidate(candidate) for candidate in get_candidate_models()]
        return {
            "status": "ok",
            "candidates": candidates,
            "active_candidate_id": get_active_candidate_id(),
            "best_candidate_id": get_best_candidate_id(),
            "warnings": [],
        }

    def get_model_options_summary(self, *, problem_type: str | None = None) -> dict[str, Any]:
        normalized_problem_type = str(problem_type).strip().lower() if problem_type else None
        if normalized_problem_type == "":
            normalized_problem_type = None
        if normalized_problem_type is None:
            normalized_problem_type = str(get_state_value("problem_type") or "").strip().lower() or None
        specs = modeling_service.list_available_model_specs(problem_type=normalized_problem_type)
        models: list[dict[str, Any]] = []
        for spec in specs:
            tuning_schema = spec.get("tuning_schema", {})
            models.append(
                {
                    "model_id": spec.get("model_id"),
                    "label": spec.get("label"),
                    "family": spec.get("family"),
                    "description": spec.get("description"),
                    "problem_types": list(spec.get("problem_types", [])),
                    "default_metrics": list(spec.get("metrics", spec.get("default_metrics", []))),
                    "param_schema": deepcopy(spec.get("param_schema", {})),
                    "tuning_schema": {
                        "supported_search_types": list(tuning_schema.get("supported_search_types", [])),
                        "default_scoring": tuning_schema.get("default_scoring"),
                        "search_space_defaults": deepcopy(tuning_schema.get("search_space_defaults", {})),
                    },
                }
            )
        return {
            "status": "ok",
            "problem_type": normalized_problem_type,
            "models": models,
            "warnings": [],
        }

    def get_model_comparison_settings_summary(self) -> dict[str, Any]:
        comparison_config = deepcopy(modeling_service.get_model_comparison_settings())
        return {
            "status": "ok",
            "comparison_settings": comparison_config,
            "warnings": [],
        }

    def update_model_comparison_settings_summary(self, **updates: object) -> dict[str, Any]:
        normalized_updates: dict[str, Any] = {}

        if "evaluation_metric" in updates:
            evaluation_metric = updates.get("evaluation_metric")
            normalized_updates["evaluation_metric"] = (
                str(evaluation_metric).strip().lower() if evaluation_metric not in (None, "") else None
            )

        if "split_strategy" in updates and updates.get("split_strategy") is not None:
            split_strategy = str(updates.get("split_strategy")).strip().lower()
            if split_strategy not in {"cross_validation", "train_test_split"}:
                return self._error(
                    "Invalid split strategy.",
                    errors=[{"field": "split_strategy", "reason": "Expected 'cross_validation' or 'train_test_split'."}],
                )
            normalized_updates["split_strategy"] = split_strategy
            normalized_updates["use_cross_validation"] = split_strategy == "cross_validation"

        if "cv_folds" in updates and updates.get("cv_folds") is not None:
            normalized_updates["cv_folds"] = int(updates.get("cv_folds"))

        if "test_size" in updates and updates.get("test_size") is not None:
            normalized_updates["test_size"] = float(updates.get("test_size"))

        if "random_seed" in updates and updates.get("random_seed") is not None:
            random_seed = int(updates.get("random_seed"))
            normalized_updates["random_seed"] = random_seed
            normalized_updates["random_state"] = random_seed

        if "classification_threshold_policy" in updates and updates.get("classification_threshold_policy") is not None:
            threshold_policy = str(updates.get("classification_threshold_policy")).strip()
            if threshold_policy not in {"Use model default", "Set manual threshold", "Optimize threshold"}:
                return self._error(
                    "Invalid classification threshold policy.",
                    errors=[
                        {
                            "field": "classification_threshold_policy",
                            "reason": "Expected 'Use model default', 'Set manual threshold', or 'Optimize threshold'.",
                        }
                    ],
                )
            normalized_updates["classification_threshold_policy"] = threshold_policy

        if "classification_threshold_manual_value" in updates and updates.get("classification_threshold_manual_value") is not None:
            normalized_updates["classification_threshold_manual_value"] = float(
                updates.get("classification_threshold_manual_value")
            )

        if "classification_threshold_objective" in updates and updates.get("classification_threshold_objective") is not None:
            threshold_objective = str(updates.get("classification_threshold_objective")).strip().title()
            if threshold_objective not in {"F1", "Precision", "Recall"}:
                return self._error(
                    "Invalid classification threshold objective.",
                    errors=[
                        {
                            "field": "classification_threshold_objective",
                            "reason": "Expected 'F1', 'Precision', or 'Recall'.",
                        }
                    ],
                )
            normalized_updates["classification_threshold_objective"] = threshold_objective

        comparison_config = modeling_service.update_model_comparison_settings(**normalized_updates)
        return {
            "status": "ok",
            "message": "Updated shared model comparison settings.",
            "changed": normalized_updates,
            "comparison_settings": deepcopy(comparison_config),
            "warnings": [],
        }

    def upsert_candidate_model(self, **updates: object) -> dict[str, Any]:
        candidate_id = str(updates.get("candidate_id") or "").strip() or None
        model_id = str(updates.get("model_id") or "").strip() or None

        if candidate_id:
            current_candidate = next(
                (candidate for candidate in get_candidate_models() if str(candidate.get("candidate_id", "")) == candidate_id),
                None,
            )
            if current_candidate is None:
                return self._error(
                    "Candidate model was not found.",
                    errors=[{"field": "candidate_id", "reason": f"Unknown candidate_id '{candidate_id}'."}],
                )
            resolved_model_id = model_id or str(current_candidate.get("model_id", ""))
            changed: dict[str, Any] = {}
            warnings: list[str] = []
            for key in (
                "candidate_label",
                "enabled",
                "train_test_split_enabled",
                "classification_threshold",
                "notes",
            ):
                if key in updates and updates.get(key) is not None:
                    changed[key] = updates.get(key)
            if model_id:
                changed["model_id"] = resolved_model_id
            if "custom_params" in updates:
                changed["custom_params"] = self._validate_custom_params(
                    model_id=resolved_model_id,
                    custom_params=updates.get("custom_params"),
                )
            if "preprocessing" in updates and updates.get("preprocessing") is not None:
                normalized_preprocessing, preprocessing_warnings = self._normalize_candidate_preprocessing_update(
                    updates.get("preprocessing")
                )
                changed["preprocessing"] = normalized_preprocessing
                warnings.extend(preprocessing_warnings)
            if "tuning" in updates and updates.get("tuning") is not None:
                changed["tuning"] = deepcopy(updates.get("tuning"))
            candidate = modeling_service.update_candidate_model_config(candidate_id, **changed)
            return {
                "status": "ok",
                "message": "Updated candidate model.",
                "candidate": self._compact_candidate(candidate),
                "warnings": warnings,
            }

        if not model_id:
            return self._error(
                "model_id is required when creating a candidate model.",
                errors=[{"field": "model_id", "reason": "This field is required."}],
            )
        candidate = modeling_service.create_candidate_model(
            model_id=model_id,
            candidate_label=str(updates.get("candidate_label")) if updates.get("candidate_label") else None,
        )
        changed: dict[str, Any] = {}
        warnings: list[str] = []
        if "enabled" in updates and updates.get("enabled") is not None:
            changed["enabled"] = bool(updates.get("enabled"))
        if "train_test_split_enabled" in updates and updates.get("train_test_split_enabled") is not None:
            changed["train_test_split_enabled"] = bool(updates.get("train_test_split_enabled"))
        if "classification_threshold" in updates and updates.get("classification_threshold") is not None:
            changed["classification_threshold"] = float(updates.get("classification_threshold"))
        if "notes" in updates and updates.get("notes") is not None:
            changed["notes"] = str(updates.get("notes"))
        if "preprocessing" in updates and updates.get("preprocessing") is not None:
            normalized_preprocessing, preprocessing_warnings = self._normalize_candidate_preprocessing_update(
                updates.get("preprocessing")
            )
            changed["preprocessing"] = normalized_preprocessing
            warnings.extend(preprocessing_warnings)
        if "tuning" in updates and updates.get("tuning") is not None:
            changed["tuning"] = deepcopy(updates.get("tuning"))
        if "custom_params" in updates:
            changed["custom_params"] = self._validate_custom_params(
                model_id=model_id,
                custom_params=updates.get("custom_params"),
            )
        if changed:
            candidate = modeling_service.update_candidate_model_config(
                str(candidate.get("candidate_id", "")),
                **changed,
            )
        return {
            "status": "ok",
            "message": "Created candidate model.",
            "candidate": self._compact_candidate(candidate),
            "warnings": warnings,
        }

    def remove_candidate_model_summary(self, candidate_id: str) -> dict[str, Any]:
        candidates_before = len(get_candidate_models())
        remaining = modeling_service.remove_candidate_model_config(candidate_id)
        if len(remaining) == candidates_before:
            return self._error(
                "Candidate model was not found.",
                errors=[{"field": "candidate_id", "reason": f"Unknown candidate_id '{candidate_id}'."}],
            )
        return {
            "status": "ok",
            "message": "Removed candidate model.",
            "removed_candidate_id": candidate_id,
            "remaining_candidate_count": len(remaining),
            "warnings": [],
        }

    def train_candidate_models_summary(
        self,
        *,
        candidate_ids: list[str] | None = None,
        only_enabled: bool | None = None,
        select_best_candidate: bool | None = None,
        set_active_stage_to_results: bool | None = None,
    ) -> dict[str, Any]:
        normalized_candidate_ids = self._normalize_column_list(candidate_ids) if candidate_ids else []
        results: list[Any] = []
        if normalized_candidate_ids:
            for candidate_id in normalized_candidate_ids:
                result = modeling_service.train_candidate_model(candidate_id)
                if result.run_record is not None:
                    modeling_service.update_candidate_model_config(
                        candidate_id,
                        latest_run_id=result.run_record.get("run_id"),
                        latest_run_record=result.run_record,
                    )
                results.append(result)
            if select_best_candidate:
                best_candidate_id = modeling_service.select_best_candidate_from_latest_results()
                from app.workspace_apps.ml_workbench.state import set_best_candidate_id

                set_best_candidate_id(best_candidate_id)
        else:
            if only_enabled is False:
                raise MLWorkbenchToolServiceError(
                    "Training all candidates regardless of enabled state is not supported in the first pass."
                )
            results = modeling_service.train_enabled_candidate_models()

        if set_active_stage_to_results:
            update_state_values(app_stage=STAGE_RESULTS)

        comparison_summary = modeling_service.build_results_comparison_summary()
        result_rows: list[dict[str, Any]] = []
        for result in results:
            run_record = result.run_record if hasattr(result, "run_record") else None
            primary_metric_name, primary_metric_value = self._primary_metric_summary(run_record if isinstance(run_record, dict) else None)
            result_rows.append(
                {
                    "candidate_id": getattr(result, "candidate_id", None),
                    "run_id": run_record.get("run_id") if isinstance(run_record, dict) else None,
                    "status": getattr(result, "status", None),
                    "primary_metric_name": primary_metric_name,
                    "primary_metric_value": primary_metric_value,
                }
            )
        return {
            "status": "ok",
            "message": f"Trained {len(result_rows)} candidate models.",
            "trained_candidate_ids": [row["candidate_id"] for row in result_rows if row.get("candidate_id")],
            "results_summary": {
                "best_candidate_id": comparison_summary.get("best_candidate_id"),
                "best_run_id": next(
                    (
                        candidate.get("latest_run_id")
                        for candidate in comparison_summary.get("candidates", [])
                        if candidate.get("candidate_id") == comparison_summary.get("best_candidate_id")
                    ),
                    None,
                ),
                "evaluation_metric": comparison_summary.get("comparison_metric_name"),
                "candidates": result_rows,
            },
            "warnings": [],
        }

    def get_results_summary(self) -> dict[str, Any]:
        summary = modeling_service.build_results_comparison_summary()
        comparison_rows: list[dict[str, Any]] = []
        for candidate in summary.get("candidates", []):
            latest_run_record = next(
                (
                    stored_candidate.get("latest_run_record")
                    for stored_candidate in get_candidate_models()
                    if str(stored_candidate.get("candidate_id", "")) == str(candidate.get("candidate_id", ""))
                ),
                None,
            )
            threshold_details: dict[str, Any] = {}
            if isinstance(latest_run_record, dict):
                if latest_run_record.get("classification_threshold_source") is not None:
                    threshold_details["classification_threshold_source"] = latest_run_record.get(
                        "classification_threshold_source"
                    )
                if latest_run_record.get("classification_threshold_used") is not None:
                    threshold_details["classification_threshold_used"] = latest_run_record.get(
                        "classification_threshold_used"
                    )
                if latest_run_record.get("classification_threshold_policy") is not None:
                    threshold_details["classification_threshold_policy"] = latest_run_record.get(
                        "classification_threshold_policy"
                    )
                if latest_run_record.get("classification_threshold_objective") is not None:
                    threshold_details["classification_threshold_objective"] = latest_run_record.get(
                        "classification_threshold_objective"
                    )
                if latest_run_record.get("classification_threshold_optimization_details") is not None:
                    threshold_details["classification_threshold_optimization_details"] = deepcopy(
                        latest_run_record.get("classification_threshold_optimization_details")
                    )
                if latest_run_record.get("cv_classification_threshold_summary") is not None:
                    threshold_details["cv_classification_threshold_summary"] = deepcopy(
                        latest_run_record.get("cv_classification_threshold_summary")
                    )
            comparison_rows.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate_label": candidate.get("candidate_label"),
                    "run_id": candidate.get("latest_run_id"),
                    "status": candidate.get("status"),
                    "primary_metric_name": candidate.get("primary_metric_name"),
                    "primary_metric_value": candidate.get("primary_metric_value"),
                    **threshold_details,
                }
            )
        return {
            "status": "ok",
            "results": {
                "best_candidate_id": summary.get("best_candidate_id"),
                "best_model_run_id": next(
                    (
                        row.get("run_id")
                        for row in comparison_rows
                        if row.get("candidate_id") == summary.get("best_candidate_id")
                    ),
                    None,
                ),
                "evaluation_metric": summary.get("comparison_metric_name"),
                "comparison_rows": comparison_rows,
            },
            "warnings": [],
        }

    def get_candidate_result_details(self, *, candidate_id: str | None = None, run_id: str | None = None) -> dict[str, Any]:
        normalized_candidate_id = str(candidate_id or "").strip() or None
        normalized_run_id = str(run_id or "").strip() or None

        selected_candidate: dict[str, Any] | None = None
        selected_run_record: dict[str, Any] | None = None

        for candidate in get_candidate_models():
            current_candidate_id = str(candidate.get("candidate_id", "")).strip()
            run_record = candidate.get("latest_run_record")
            if not isinstance(run_record, dict):
                continue
            current_run_id = str(run_record.get("run_id", "")).strip()
            if normalized_run_id and current_run_id == normalized_run_id:
                selected_candidate = candidate
                selected_run_record = run_record
                break
            if normalized_candidate_id and current_candidate_id == normalized_candidate_id:
                selected_candidate = candidate
                selected_run_record = run_record
                break

        if selected_candidate is None or selected_run_record is None:
            return self._error(
                "Candidate result details were not found.",
                errors=[
                    {
                        "field": "candidate_id" if normalized_candidate_id else "run_id",
                        "reason": "No matching latest run record was found.",
                    }
                ],
            )

        metrics = deepcopy(selected_run_record.get("metrics", {}))
        result_details = {
            "candidate_id": str(selected_candidate.get("candidate_id", "")),
            "candidate_label": str(selected_candidate.get("candidate_label", "")),
            "model_id": str(selected_candidate.get("model_id", "")),
            "run_id": selected_run_record.get("run_id"),
            "status": selected_run_record.get("status"),
            "training_mode": selected_run_record.get("training_mode"),
            "split_strategy": selected_run_record.get("split_strategy"),
            "evaluation_metric": modeling_service.get_model_comparison_settings().get("evaluation_metric"),
            "metrics": metrics,
            "classification_threshold": {
                "source": selected_run_record.get("classification_threshold_source"),
                "policy": selected_run_record.get("classification_threshold_policy"),
                "objective": selected_run_record.get("classification_threshold_objective"),
                "used": selected_run_record.get("classification_threshold_used"),
                "manual_value": selected_run_record.get("classification_threshold_manual_value"),
                "optimization_details": deepcopy(
                    selected_run_record.get("classification_threshold_optimization_details")
                ),
                "cv_summary": deepcopy(selected_run_record.get("cv_classification_threshold_summary")),
            },
            "preprocessing_summary": deepcopy(selected_run_record.get("preprocessing_summary", {})),
            "params_used": deepcopy(selected_run_record.get("params_used", {})),
            "tuning_result": deepcopy(selected_run_record.get("tuning_result")),
            "feature_columns": list(selected_run_record.get("feature_columns", [])),
            "notes": selected_run_record.get("notes"),
        }
        return {
            "status": "ok",
            "result": result_details,
            "warnings": [],
        }
