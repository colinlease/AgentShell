

"""Dataset loading and artifact-building helpers for the ML Workbench app."""

from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from app.workspace_apps.ml_workbench.constants import (
    APP_ID,
    ARTIFACT_KIND_DATASET,
    ARTIFACT_MODEL_INPUT_DATASET,
    ARTIFACT_RAW_DATASET,
    ARTIFACT_ROLE_MODEL_INPUT,
    ARTIFACT_ROLE_RAW,
    ARTIFACT_ROLE_WORKING,
    ARTIFACT_WORKING_DATASET,
    STAGE_PREPROCESS,
    STAGE_UPLOAD,
)
from app.workspace_apps.ml_workbench.schemas import ArtifactMetadata
from app.workspace_apps.ml_workbench.services.artifact_service import (
    get_artifact,
    get_artifact_object,
    register_artifact,
    update_artifact,
)
from app.workspace_apps.ml_workbench.state import (
    get_app_state,
    reset_app_state,
    set_state_value,
    update_status_flags,
)

SUPPORTED_FILE_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class DatasetLoadError(ValueError):
    """Raised when a dataset cannot be loaded into the ML Workbench."""



def _normalize_uploaded_file_name(uploaded_file: Any) -> str:
    """Return a readable file name for an uploaded file-like object."""
    name = getattr(uploaded_file, "name", None)
    if isinstance(name, str) and name.strip():
        return Path(name).name
    return "uploaded_dataset"



def _detect_extension(file_name: str) -> str:
    """Return the lowercase file extension for the supplied file name."""
    return Path(file_name).suffix.lower()



def validate_file_extension(file_name: str) -> None:
    """Raise an error when the file extension is unsupported."""
    extension = _detect_extension(file_name)
    if extension not in SUPPORTED_FILE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_FILE_EXTENSIONS))
        raise DatasetLoadError(
            f"Unsupported file type '{extension or '[none]'}'. Supported types: {supported}."
        )



def validate_dataframe(df: pd.DataFrame) -> None:
    """Raise an error when the dataframe is not suitable for the app."""
    if not isinstance(df, pd.DataFrame):
        raise DatasetLoadError("Loaded object is not a pandas DataFrame.")
    if df.empty:
        raise DatasetLoadError("The uploaded dataset is empty.")
    if df.shape[1] == 0:
        raise DatasetLoadError("The uploaded dataset has no columns.")
    if df.columns.duplicated().any():
        duplicate_columns = [str(col) for col in df.columns[df.columns.duplicated()].tolist()]
        raise DatasetLoadError(
            "The uploaded dataset contains duplicate column names: "
            + ", ".join(duplicate_columns)
        )



def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read a CSV file from bytes."""
    try:
        return pd.read_csv(BytesIO(file_bytes))
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(file_bytes), encoding="latin-1")
    except Exception as exc:  # pragma: no cover - defensive branch
        raise DatasetLoadError(f"Unable to read CSV file: {exc}") from exc



def _read_excel(file_bytes: bytes) -> pd.DataFrame:
    """Read an Excel file from bytes."""
    try:
        return pd.read_excel(BytesIO(file_bytes))
    except Exception as exc:  # pragma: no cover - defensive branch
        raise DatasetLoadError(f"Unable to read Excel file: {exc}") from exc



def dataframe_from_uploaded_file(uploaded_file: Any) -> tuple[pd.DataFrame, str]:
    """Load a pandas DataFrame from a Streamlit-uploaded file object.

    Returns the dataframe and the normalized file name.
    """
    if uploaded_file is None:
        raise DatasetLoadError("No file was provided.")

    file_name = _normalize_uploaded_file_name(uploaded_file)
    validate_file_extension(file_name)
    extension = _detect_extension(file_name)

    try:
        file_bytes = uploaded_file.getvalue()
    except Exception as exc:  # pragma: no cover - defensive branch
        raise DatasetLoadError(f"Unable to read uploaded file bytes: {exc}") from exc

    if not isinstance(file_bytes, (bytes, bytearray)) or len(file_bytes) == 0:
        raise DatasetLoadError("The uploaded file is empty.")

    if extension == ".csv":
        df = _read_csv(bytes(file_bytes))
    else:
        df = _read_excel(bytes(file_bytes))

    validate_dataframe(df)
    return df, file_name



def _build_dataset_metadata(
    df: pd.DataFrame,
    *,
    note: str,
    source_file_name: str | None,
    created_from_stage: str,
    target_column: str | None = None,
    problem_type: str | None = None,
    ready_for_modeling: bool = False,
) -> ArtifactMetadata:
    """Build standardized metadata for a dataframe artifact."""
    missing = df.isna().sum()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtype_summary": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_summary": {
            column: int(count) for column, count in missing.items() if int(count) > 0
        },
        "target_column": target_column,
        "problem_type": problem_type,
        "note": note,
        "source_file_name": source_file_name,
        "created_from_stage": created_from_stage,
        "ready_for_modeling": ready_for_modeling,
    }



def register_raw_dataset(df: pd.DataFrame, *, file_name: str) -> pd.DataFrame:
    """Register the uploaded dataframe as the authoritative raw dataset."""
    validate_dataframe(df)
    register_artifact(
        name=ARTIFACT_RAW_DATASET,
        kind=ARTIFACT_KIND_DATASET,
        role=ARTIFACT_ROLE_RAW,
        obj=df,
        metadata=_build_dataset_metadata(
            df,
            note="Original uploaded dataset.",
            source_file_name=file_name,
            created_from_stage=STAGE_UPLOAD,
            ready_for_modeling=False,
        ),
    )
    return df



def create_working_dataset(*, source_name: str = ARTIFACT_RAW_DATASET) -> pd.DataFrame:
    """Create the working dataset as a copy of the specified source dataset."""
    source_df = get_dataset(source_name)
    working_df = source_df.copy(deep=True)
    source_record = get_artifact(source_name)
    source_file_name = None
    if source_record is not None:
        source_file_name = source_record["metadata"].get("source_file_name")

    register_artifact(
        name=ARTIFACT_WORKING_DATASET,
        kind=ARTIFACT_KIND_DATASET,
        role=ARTIFACT_ROLE_WORKING,
        obj=working_df,
        source_artifact=source_name,
        metadata=_build_dataset_metadata(
            working_df,
            note="Working dataset derived from the raw upload.",
            source_file_name=source_file_name,
            created_from_stage=STAGE_PREPROCESS,
            ready_for_modeling=False,
        ),
    )
    return working_df



def set_working_dataset(df: pd.DataFrame, *, source_name: str = ARTIFACT_RAW_DATASET) -> pd.DataFrame:
    """Replace the working dataset artifact with the supplied dataframe."""
    validate_dataframe(df)
    source_record = get_artifact(source_name)
    source_file_name = None
    if source_record is not None:
        source_file_name = source_record["metadata"].get("source_file_name")

    if get_artifact(ARTIFACT_WORKING_DATASET) is None:
        register_artifact(
            name=ARTIFACT_WORKING_DATASET,
            kind=ARTIFACT_KIND_DATASET,
            role=ARTIFACT_ROLE_WORKING,
            obj=df,
            source_artifact=source_name,
            metadata=_build_dataset_metadata(
                df,
                note="Working dataset for preprocessing and feature engineering.",
                source_file_name=source_file_name,
                created_from_stage=STAGE_PREPROCESS,
                ready_for_modeling=False,
            ),
        )
    else:
        update_artifact(
            ARTIFACT_WORKING_DATASET,
            obj=df,
            role=ARTIFACT_ROLE_WORKING,
            source_artifact=source_name,
            metadata_updates={
                "note": "Working dataset for preprocessing and feature engineering.",
                "source_file_name": source_file_name,
                "created_from_stage": STAGE_PREPROCESS,
                "ready_for_modeling": False,
            },
        )
    return df



def build_model_input_dataset(
    *,
    source_name: str = ARTIFACT_WORKING_DATASET,
    target_column: str | None = None,
    problem_type: str | None = None,
    selected_feature_columns: list[str] | None = None,
    ignored_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Create a model-input dataset from the working dataset.

    The returned dataframe is intentionally still a single dataframe artifact.
    Train/test splitting and model-specific transformations can happen later.
    """
    source_df = get_dataset(source_name)
    model_df = source_df.copy(deep=True)

    if ignored_columns:
        drop_candidates = [column for column in ignored_columns if column in model_df.columns]
        if drop_candidates:
            model_df = model_df.drop(columns=drop_candidates)

    if selected_feature_columns:
        required_columns = list(selected_feature_columns)
        if target_column and target_column not in required_columns:
            required_columns.append(target_column)
        missing_columns = [column for column in required_columns if column not in model_df.columns]
        if missing_columns:
            raise DatasetLoadError(
                "One or more selected feature columns are not present in the dataset: "
                + ", ".join(missing_columns)
            )
        model_df = model_df[required_columns].copy()

    validate_dataframe(model_df)

    source_record = get_artifact(source_name)
    source_file_name = None
    if source_record is not None:
        source_file_name = source_record["metadata"].get("source_file_name")

    register_artifact(
        name=ARTIFACT_MODEL_INPUT_DATASET,
        kind=ARTIFACT_KIND_DATASET,
        role=ARTIFACT_ROLE_MODEL_INPUT,
        obj=model_df,
        source_artifact=source_name,
        metadata=_build_dataset_metadata(
            model_df,
            note="Model-input dataset derived from the working dataset.",
            source_file_name=source_file_name,
            created_from_stage=STAGE_PREPROCESS,
            target_column=target_column,
            problem_type=problem_type,
            ready_for_modeling=True,
        ),
    )
    return model_df



def get_dataset(dataset_name: str = ARTIFACT_RAW_DATASET) -> pd.DataFrame:
    """Return one named dataset artifact as a dataframe."""
    dataset = get_artifact_object(dataset_name)
    if dataset is None:
        raise DatasetLoadError(f"Dataset artifact '{dataset_name}' was not found.")
    if not isinstance(dataset, pd.DataFrame):
        raise DatasetLoadError(
            f"Artifact '{dataset_name}' is not a dataframe and cannot be used as a dataset."
        )
    return dataset



def get_dataset_copy(dataset_name: str = ARTIFACT_RAW_DATASET) -> pd.DataFrame:
    """Return a deep copy of one named dataset artifact."""
    return get_dataset(dataset_name).copy(deep=True)



def load_uploaded_dataset(uploaded_file: Any) -> pd.DataFrame:
    """Load an uploaded dataset and initialize the core dataset artifacts.

    This is the main entry point for the app's data-loading flow.
    """
    df, file_name = dataframe_from_uploaded_file(uploaded_file)

    reset_app_state(preserve_loaded_file_name=False)
    register_raw_dataset(df, file_name=file_name)
    create_working_dataset(source_name=ARTIFACT_RAW_DATASET)

    set_state_value("loaded_file_name", file_name)
    set_state_value("active_dataset_name", ARTIFACT_WORKING_DATASET)
    update_status_flags(
        dataset_loaded=True,
        profile_ready=True,
        preprocessing_applied=False,
        features_applied=False,
        model_input_ready=False,
        models_trained=False,
        results_ready=False,
    )
    return df



def sync_dataset_metadata_from_state() -> None:
    """Sync dataset artifact metadata with the current app state.

    This keeps published artifact context aligned with selected problem type,
    target column, and active modeling readiness.
    """
    state = get_app_state()
    target_column = state.get("target_column")
    problem_type = state.get("problem_type")
    selected_feature_columns = state.get("selected_feature_columns", [])
    ignored_columns = state.get("ignored_columns", [])

    for artifact_name in [
        ARTIFACT_RAW_DATASET,
        ARTIFACT_WORKING_DATASET,
        ARTIFACT_MODEL_INPUT_DATASET,
    ]:
        record = get_artifact(artifact_name)
        if record is None:
            continue

        ready_for_modeling = artifact_name == ARTIFACT_MODEL_INPUT_DATASET
        note = record["metadata"].get("note", "")
        if artifact_name == ARTIFACT_MODEL_INPUT_DATASET:
            note = "Model-input dataset derived from the working dataset."
        elif artifact_name == ARTIFACT_WORKING_DATASET and selected_feature_columns:
            note = "Working dataset with preprocessing and feature engineering applied."
        elif artifact_name == ARTIFACT_RAW_DATASET:
            note = "Original uploaded dataset."

        metadata_updates: dict[str, Any] = {
            "target_column": target_column,
            "problem_type": problem_type,
            "ready_for_modeling": ready_for_modeling,
            "note": note,
        }

        if artifact_name == ARTIFACT_MODEL_INPUT_DATASET:
            metadata_updates["selected_feature_columns"] = deepcopy(selected_feature_columns)
            metadata_updates["ignored_columns"] = deepcopy(ignored_columns)

        update_artifact(artifact_name, metadata_updates=metadata_updates)



def get_available_dataset_names() -> list[str]:
    """Return dataset artifact names in their expected workflow order."""
    ordered_names = [
        ARTIFACT_RAW_DATASET,
        ARTIFACT_WORKING_DATASET,
        ARTIFACT_MODEL_INPUT_DATASET,
    ]
    return [name for name in ordered_names if get_artifact(name) is not None]



def get_active_dataset_name(default_to_working: bool = True) -> str | None:
    """Return the current active dataset name from app state.

    When `default_to_working` is True, the function falls back to the working
    dataset, then the raw dataset, if the stored active dataset name is absent.
    """
    state = get_app_state()
    active_name = state.get("active_dataset_name")
    if active_name and get_artifact(active_name) is not None:
        return active_name
    if not default_to_working:
        return None
    if get_artifact(ARTIFACT_WORKING_DATASET) is not None:
        return ARTIFACT_WORKING_DATASET
    if get_artifact(ARTIFACT_RAW_DATASET) is not None:
        return ARTIFACT_RAW_DATASET
    return None



def set_active_dataset_name(dataset_name: str) -> None:
    """Set the active dataset name after validating that it exists."""
    if get_artifact(dataset_name) is None:
        raise DatasetLoadError(f"Dataset artifact '{dataset_name}' was not found.")
    set_state_value("active_dataset_name", dataset_name)



def get_dataset_preview(dataset_name: str | None = None, row_limit: int = 50) -> pd.DataFrame:
    """Return a UI-friendly preview of a dataset artifact."""
    resolved_name = dataset_name or get_active_dataset_name()
    if resolved_name is None:
        raise DatasetLoadError("No dataset is currently available for preview.")
    df = get_dataset(resolved_name)
    safe_row_limit = max(1, int(row_limit))
    return df.head(safe_row_limit).copy()



def dataset_summary(dataset_name: str) -> dict[str, Any]:
    """Return a compact summary for one dataset artifact."""
    record = get_artifact(dataset_name)
    if record is None:
        raise DatasetLoadError(f"Dataset artifact '{dataset_name}' was not found.")
    metadata = record["metadata"]
    return {
        "name": record["name"],
        "kind": record["kind"],
        "role": record["role"],
        "rows": metadata.get("rows", 0),
        "columns": metadata.get("columns", 0),
        "column_names": metadata.get("column_names", []),
        "dtype_summary": metadata.get("dtype_summary", {}),
        "missing_summary": metadata.get("missing_summary", {}),
        "target_column": metadata.get("target_column"),
        "problem_type": metadata.get("problem_type"),
        "ready_for_modeling": metadata.get("ready_for_modeling", False),
        "note": metadata.get("note", ""),
        "source_file_name": metadata.get("source_file_name"),
    }



def list_dataset_summaries() -> list[dict[str, Any]]:
    """Return summaries for all available dataset artifacts."""
    return [dataset_summary(name) for name in get_available_dataset_names()]



def get_workspace_dataset_object(dataset_name: str | None = None) -> pd.DataFrame | None:
    """Return the dataset object exposed to AgentShell.

    This is the clean dataset-access surface that shell-level tools can use.
    If a dataset name is supplied, that named dataset is returned. Otherwise the
    current active dataset is used, falling back to the working dataset and then
    the raw dataset.
    """
    resolved_name = dataset_name or get_active_dataset_name(default_to_working=True)
    if resolved_name is None:
        return None
    return get_artifact_object(resolved_name)



def has_loaded_dataset() -> bool:
    """Return True when the raw dataset artifact exists."""
    return get_artifact(ARTIFACT_RAW_DATASET) is not None