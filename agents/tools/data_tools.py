from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool
from domain.services.dataset_feature_engineering_service import DatasetFeatureEngineeringService
from domain.services.data_tools_service import DataToolsService


class GetDatasetProfileTool(BaseTool):
    """
    Read-only tool for returning a deeper analysis-oriented profile of one
    currently loaded dataset.

    This tool is intentionally narrower than `get_loaded_data_context()`. The
    context tool answers what datasets exist. This tool answers what is
    important about a specific dataset for EDA / ML reasoning, including
    compact low-cardinality value distributions and optional focus on one
    specific column.
    """

    name = "get_dataset_profile"
    description = "Return a structured analysis-oriented profile for one loaded dataset, including missingness detail, per-column completeness, distinct counts, duplicate-row count, constant columns, numeric summary statistics, compact low-cardinality value distributions, and optional focus on one specific column."
    category = "data"
    scope = "framework"
    is_read_only = True
    is_enabled_by_default = True
    permission_level = "standard"
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "dataset_name": {
                "type": ["string", "null"],
                "description": "Optional dataset name. If omitted, profile the active dataset.",
            },
            "focus_column": {
                "type": ["string", "null"],
                "description": "Optional column to focus the profile on. If provided, return deeper profile detail primarily for this column instead of all columns.",
            },
        },
        "required": [],
    }

    def __init__(self, data_tools_service: DataToolsService | None = None) -> None:
        self.data_tools_service = data_tools_service or DataToolsService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Return a structured profile for one loaded dataset.
        """
        dataset_name = kwargs.get("dataset_name")
        dataset_name = str(dataset_name) if dataset_name not in (None, "") else None

        focus_column = kwargs.get("focus_column")
        focus_column = str(focus_column) if focus_column not in (None, "") else None

        return self.data_tools_service.get_dataset_profile(
            dataset_name=dataset_name,
            focus_column=focus_column,
        )


class GetDatasetSampleTool(BaseTool):
    """
    Read-only tool for returning a compact but meaningful sample of one
    currently loaded dataset.

    This tool is intended to let the agent inspect representative real rows
    without pulling an unnecessarily large payload into the conversation.
    """

    name = "get_dataset_sample"
    description = "Return a compact row sample for one loaded dataset so the agent can inspect representative values without retrieving the full dataset."
    category = "data"
    scope = "framework"
    is_read_only = True
    is_enabled_by_default = True
    permission_level = "standard"
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "dataset_name": {
                "type": ["string", "null"],
                "description": "Optional dataset name. If omitted, sample the active dataset.",
            },
            "sample_type": {
                "type": ["string", "null"],
                "description": "Optional sample type. Supported values are 'head' and 'random'. Defaults to 'head'.",
                "enum": ["head", "random", None],
            },
            "row_count": {
                "type": ["integer", "null"],
                "description": "Optional requested row count. The service will cap this to a small meaningful maximum.",
                "minimum": 1,
                "maximum": 25,
            },
        },
        "required": [],
    }

    def __init__(self, data_tools_service: DataToolsService | None = None) -> None:
        self.data_tools_service = data_tools_service or DataToolsService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Return a compact sample for one loaded dataset.
        """
        dataset_name = kwargs.get("dataset_name")
        dataset_name = str(dataset_name) if dataset_name not in (None, "") else None

        sample_type = kwargs.get("sample_type")
        sample_type = str(sample_type).strip().lower() if sample_type not in (None, "") else "head"
        if sample_type not in {"head", "random"}:
            sample_type = "head"

        row_count = kwargs.get("row_count")
        try:
            row_count_value = int(row_count) if row_count not in (None, "") else 10
        except (TypeError, ValueError):
            row_count_value = 10

        return self.data_tools_service.get_dataset_sample(
            dataset_name=dataset_name,
            sample_type=sample_type,
            row_count=row_count_value,
        )


class DeriveDatasetFeaturesTool(BaseTool):
    """
    Create a session-scoped derived dataset with appended engineered columns.

    This tool does not mutate app-owned datasets or source files. A successful
    call makes the derived dataset the active framework dataset for later
    general data tools when dataset_name is omitted.
    """

    name = "derive_dataset_features"
    description = (
        "Create a safe session-only derived dataset copy with engineered feature columns appended. "
        "Use this before sampling, profiling, aggregating, or charting calculations that need new columns, "
        "such as A minus B or numeric flags. The source app dataset and source files are not mutated. "
        "On success, the derived dataset becomes the active framework dataset, so future general dataset "
        "tools default to the derived dataset when dataset_name is omitted. Supports operations: add, "
        "subtract, multiply, divide, ratio, log, log1p, square, cube, flag_gt, flag_gte, flag_lt, "
        "flag_lte, flag_eq, flag_is_missing, and datetime_part."
    )
    category = "data"
    scope = "framework"
    is_read_only = False
    is_enabled_by_default = True
    permission_level = "standard"
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "dataset_name": {
                "type": ["string", "null"],
                "description": "Optional source dataset name. If omitted, use the current framework-active dataset.",
            },
            "output_dataset_name": {
                "type": ["string", "null"],
                "description": "Optional name for the derived dataset. If omitted, a stable derived name is generated.",
            },
            "features": {
                "type": "array",
                "description": "Feature specifications to append to the derived dataset. At most 10 features are supported per call.",
                "minItems": 1,
                "maxItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "New feature column name. Must not already exist in the source dataset.",
                        },
                        "operation": {
                            "type": "string",
                            "description": "Feature operation to apply.",
                            "enum": [
                                "add", "subtract", "multiply", "divide", "ratio",
                                "log", "log1p", "square", "cube",
                                "flag_gt", "flag_gte", "flag_lt", "flag_lte", "flag_eq", "flag_is_missing",
                                "datetime_part",
                            ],
                        },
                        "source_columns": {
                            "type": "array",
                            "description": "Existing or previously-created feature columns used by this feature.",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 2,
                        },
                        "parameters": {
                            "type": ["object", "null"],
                            "description": "Optional operation parameters. Flags use threshold or compare_value. datetime_part uses part: year, quarter, month, week, day, dayofweek, or hour.",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["name", "operation", "source_columns"],
                    "additionalProperties": False,
                },
            },
            "preview_rows": {
                "type": ["integer", "null"],
                "description": "Optional preview row count returned by the tool. Defaults to 10 and is capped at 20.",
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["features"],
        "additionalProperties": False,
    }

    def __init__(self, feature_service: DatasetFeatureEngineeringService | None = None) -> None:
        self.feature_service = feature_service or DatasetFeatureEngineeringService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        dataset_name = kwargs.get("dataset_name")
        dataset_name = str(dataset_name) if dataset_name not in (None, "") else None

        output_dataset_name = kwargs.get("output_dataset_name")
        output_dataset_name = str(output_dataset_name) if output_dataset_name not in (None, "") else None

        features = kwargs.get("features")
        features_value = features if isinstance(features, list) else None

        preview_rows = kwargs.get("preview_rows")
        try:
            preview_rows_value = int(preview_rows) if preview_rows not in (None, "") else 10
        except (TypeError, ValueError):
            preview_rows_value = 10

        return self.feature_service.derive_dataset_features(
            dataset_name=dataset_name,
            output_dataset_name=output_dataset_name,
            features=features_value,
            preview_rows=preview_rows_value,
        )


class GetDatasetAggregationTool(BaseTool):
    """
    Read-only tool for returning a compact grouped / filtered aggregation of one
    currently loaded dataset.

    This tool is intended for targeted analytical questions such as comparing
    counts, totals, or averages across categories or time periods without
    retrieving raw rows. IMPORTANT: at most 3 aggregation metrics are supported
    in a single call. For metrics, use operation='count' without target_column
    for filtered row counts, or use operation='count' with target_column for
    non-null counts of a specific column.
    """

    name = "get_dataset_aggregation"
    description = "Return a compact grouped / filtered aggregation for one loaded dataset, including optional filters, grouping columns, explicit aggregation metrics, optional time bucketing for date or datetime columns, sorting, and row limits. Important: at most 3 aggregation metrics are supported in a single call. Use operation='count' without target_column for filtered row counts, or use operation='count' with target_column for non-null counts of a specific column."
    category = "data"
    scope = "framework"
    is_read_only = True
    is_enabled_by_default = True
    permission_level = "standard"
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "dataset_name": {
                "type": ["string", "null"],
                "description": "Optional dataset name. If omitted, aggregate the active dataset.",
            },
            "filters": {
                "type": ["array", "null"],
                "description": "Optional list of row filters. All filters are combined with AND.",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "Column name to filter on.",
                        },
                        "operator": {
                            "type": "string",
                            "description": "Comparison operator.",
                            "enum": [
                                "==", "!=", ">", ">=", "<", "<=",
                                "in", "not_in",
                                "contains", "not_contains",
                                "is_null", "is_not_null",
                            ],
                        },
                        "value": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "integer"},
                                {"type": "boolean"},
                                {
                                    "type": "array",
                                    "items": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "number"},
                                            {"type": "integer"},
                                            {"type": "boolean"},
                                            {"type": "null"},
                                        ],
                                    },
                                },
                                {"type": "null"},
                            ],
                            "description": "Optional comparison value. Use an array for 'in' and 'not_in'. Omit for 'is_null' and 'is_not_null'.",
                        },
                    },
                    "required": ["column", "operator"],
                },
            },
            "group_by": {
                "type": ["array", "null"],
                "description": "Optional list of existing dataset columns to group by.",
                "items": {
                    "type": "string",
                },
            },
            "time_bucket": {
                "type": ["object", "null"],
                "description": "Optional time bucketing instruction for a date or datetime column.",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Date or datetime column to parse and bucket.",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Time unit to aggregate by.",
                        "enum": ["year", "quarter", "month", "week", "day", "hour"],
                    },
                    "label": {
                        "type": ["string", "null"],
                        "description": "Optional output column name for the derived time bucket.",
                    },
                },
                "required": ["column", "unit"],
            },
            "metrics": {
                "type": "array",
                "description": "List of aggregation metrics to compute. At most 3 metrics are supported per call. Use operation='count' without target_column for filtered row counts. Use operation='count' with target_column for non-null counts of a specific column.",
                "items": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "description": "Aggregation function.",
                            "enum": ["count", "sum", "mean", "median", "min", "max", "nunique"],
                        },
                        "target_column": {
                            "type": ["string", "null"],
                            "description": "Column to aggregate. For operation='count', omit target_column to count filtered rows, or provide target_column to count non-null values in that column.",
                        },
                        "alias": {
                            "type": ["string", "null"],
                            "description": "Optional output name for this metric.",
                        },
                    },
                    "required": ["operation"],
                },
            },
            "sort_by": {
                "type": ["string", "null"],
                "description": "Optional output column name to sort by.",
            },
            "sort_direction": {
                "type": ["string", "null"],
                "description": "Optional sort direction.",
                "enum": ["asc", "desc", None],
            },
            "limit": {
                "type": ["integer", "null"],
                "description": "Optional maximum number of result rows to return. The service will cap this to a compact maximum.",
                "minimum": 1,
                "maximum": 100,
            },
        },
        "required": ["metrics"],
    }

    def __init__(self, data_tools_service: DataToolsService | None = None) -> None:
        self.data_tools_service = data_tools_service or DataToolsService()

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """
        Return a compact grouped / filtered aggregation for one loaded dataset.
        """
        dataset_name = kwargs.get("dataset_name")
        dataset_name = str(dataset_name) if dataset_name not in (None, "") else None

        filters = kwargs.get("filters")
        filters_value = filters if isinstance(filters, list) else None

        group_by = kwargs.get("group_by")
        group_by_value = [str(column) for column in group_by] if isinstance(group_by, list) else None

        time_bucket = kwargs.get("time_bucket")
        time_bucket_value = time_bucket if isinstance(time_bucket, dict) else None

        metrics = kwargs.get("metrics")
        metrics_value = metrics if isinstance(metrics, list) else []

        sort_by = kwargs.get("sort_by")
        sort_by_value = str(sort_by) if sort_by not in (None, "") else None

        sort_direction = kwargs.get("sort_direction")
        sort_direction_value = str(sort_direction).strip().lower() if sort_direction not in (None, "") else None
        if sort_direction_value not in {None, "asc", "desc"}:
            sort_direction_value = None

        limit = kwargs.get("limit")
        try:
            limit_value = int(limit) if limit not in (None, "") else None
        except (TypeError, ValueError):
            limit_value = None

        return self.data_tools_service.get_dataset_aggregation(
            dataset_name=dataset_name,
            filters=filters_value,
            group_by=group_by_value,
            time_bucket=time_bucket_value,
            metrics=metrics_value,
            sort_by=sort_by_value,
            sort_direction=sort_direction_value,
            limit=limit_value,
        )
