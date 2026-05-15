from __future__ import annotations

from typing import Any

from agents.tools.base import BaseTool
from domain.services.visualization_tools_service import VisualizationToolsService


_FILTER_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": "Optional row filters to apply before building the chart.",
    "items": {
        "type": "object",
        "properties": {
            "column": {
                "type": "string",
                "description": "Dataset column to filter on.",
            },
            "operator": {
                "type": "string",
                "description": "Comparison operator.",
                "enum": [
                    "==",
                    "!=",
                    ">",
                    ">=",
                    "<",
                    "<=",
                    "contains",
                    "starts_with",
                    "ends_with",
                    "in",
                    "not_in",
                    "is_null",
                    "is_not_null",
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
                            ]
                        },
                    },
                    {"type": "null"},
                ],
                "description": "Optional comparison value. Use an array for 'in' and 'not_in'. Omit for 'is_null' and 'is_not_null'.",
            },
        },
        "required": ["column", "operator"],
        "additionalProperties": False,
    },
}


class GetDatasetChartTool(BaseTool):
    """
    Build one standardized chart from a loaded dataset for inline display in chat.

    Use this for a single quick exploratory chart. Supported chart types are
    bar, line, scatter, and histogram.

    Chart-specific guidance:
    - bar: compare grouped categories. x_column is required. y_column is optional.
      aggregation is optional when y_column is provided. limit is useful for
      keeping category counts readable.
    - line: show a trend over time or across an ordered x-axis. x_column is
      required and y_column is usually required. Use time_bucket only when
      x_column is a date/datetime field and the series is dense.
    - scatter: show the relationship between two numeric fields. Requires both
      x_column and y_column. Do not provide aggregation, time_bucket, or bins.
      Omit limit unless you intentionally want a specific sample size Maximum. Auto-sampled if over 1,000 points. 
    - histogram: show the distribution of one numeric field. Requires x_column
      only. Do not provide y_column, aggregation, or time_bucket. Maximum of 20 bins.

    Important:
    - Request only one chart per call.
    - Omit arguments that do not apply to the chosen chart type.
    - Keep charts readable by limiting bars/bins/points.
    """

    name = "get_dataset_chart"
    description = (
        "Build one chart from a loaded dataset for inline display in chat. "
        "Use this when the user asks to chart, plot, graph, or visualize data. "
        "Supported chart types: bar, line, scatter, histogram. Request only one chart per call. "
        "Use only arguments that apply to the chosen chart type. "
        "Bar: x_column required, y_column optional, aggregation optional, limit useful for readable category labels. "
        "Line: use for trends; y_column required; use time_bucket only for date/datetime x-columns when the series is dense. "
        "Scatter: requires x_column and y_column; both should be numeric; do not provide aggregation, time_bucket, or bins; omit limit unless you intentionally want a specific sample size. Auto-enforced maximum of 500 points."
        "Histogram: use x_column only; do not provide y_column, aggregation, or time_bucket; bins applies only to histograms. "
        "Use only real dataset columns in `working_dataset`; upsert required features before building the chart. "
        "Keep charts readable by limiting bars, bins, and points."
    )
    category = "visualization"
    scope = "framework"
    is_read_only = True
    is_enabled_by_default = True
    permission_level = "standard"
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "dataset_name": {
                "type": "string",
                "description": "Optional dataset name. If omitted, the active dataset is used.",
            },
            "chart_type": {
                "type": "string",
                "description": "Chart type to build.",
                "enum": ["bar", "line", "scatter", "histogram"],
            },
            "x_column": {
                "type": "string",
                "description": "Primary x-axis column. Required for all chart types.",
            },
            "y_column": {
                "type": "string",
                "description": "Optional y-axis column. Required for line and scatter charts. Optional for bar charts. Do not provide for histograms.",
            },
            "aggregation": {
                "type": "string",
                "description": "Optional aggregation for bar or line charts when y_column is provided. Do not provide for scatter or histogram charts. If omitted for bar charts, the tool may count rows by category.",
                "enum": ["count", "sum", "mean", "median", "min", "max"],
            },
            "filters": _FILTER_SCHEMA,
            "time_bucket": {
                "type": "string",
                "description": "Optional time bucketing for line charts when x_column is a date or datetime field. Do not provide for bar, scatter, or histogram charts.",
                "enum": ["day", "week", "month", "quarter", "year"],
            },
            "bins": {
                "type": "integer",
                "description": "Optional number of bins for histogram charts only. Do not provide for bar, line, or scatter charts.",
                "minimum": 3,
                "maximum": 20,
            },
            "limit": {
                "type": "integer",
                "description": "Optional cap on plotted categories or points. Mainly useful for bar charts and for scatter only when you intentionally want a specific sample size. Omit unless needed.",
                "minimum": 1,
                "maximum": 1000,
            },
            "title": {
                "type": "string",
                "description": "Optional chart title to display in chat.",
            },
        },
        "required": ["chart_type", "x_column"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.visualization_tools_service = VisualizationToolsService()

    def run(self, **kwargs: Any) -> Any:
        dataset_name = kwargs.get("dataset_name")
        chart_type = str(kwargs.get("chart_type", "")).strip().lower()
        x_column = str(kwargs.get("x_column", "")).strip()
        y_column_raw = kwargs.get("y_column")
        y_column = str(y_column_raw).strip() if y_column_raw is not None else None
        if y_column == "":
            y_column = None

        aggregation_raw = kwargs.get("aggregation")
        aggregation = (
            str(aggregation_raw).strip().lower() if aggregation_raw is not None else None
        )
        if aggregation == "":
            aggregation = None

        filters = kwargs.get("filters") or None
        time_bucket_raw = kwargs.get("time_bucket")
        time_bucket = (
            str(time_bucket_raw).strip().lower() if time_bucket_raw is not None else None
        )
        if time_bucket == "":
            time_bucket = None

        bins = kwargs.get("bins")
        limit = kwargs.get("limit")
        title_raw = kwargs.get("title")
        title = str(title_raw).strip() if title_raw is not None else None
        if title == "":
            title = None

        return self.visualization_tools_service.get_dataset_chart(
            dataset_name=dataset_name,
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            aggregation=aggregation,
            filters=filters,
            time_bucket=time_bucket,
            bins=bins,
            limit=limit,
            title=title,
        )
