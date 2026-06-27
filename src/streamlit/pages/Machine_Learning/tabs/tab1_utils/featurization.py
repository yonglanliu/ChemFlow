# Copyright (c) 2026 Yonglan Liu
# Licensed under the MIT License.

from pathlib import Path
import json
import pickle
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st

from src.streamlit.utils.design import temp_error, temp_info, temp_success
from src.chemflow.machine_learning.data import MOL_REP_NAMES


# ============================================================
# File loading
# ============================================================

def load_file(file_path):
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(file_path)

    if ext == ".tsv":
        return pd.read_csv(file_path, sep="\t")

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if ext in [".pkl", ".pickle"]:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    if ext == ".parquet":
        return pd.read_parquet(file_path)

    if ext == ".feather":
        return pd.read_feather(file_path)

    if ext in [".db", ".sqlite", ".sqlite3"]:
        return sqlite3.connect(str(file_path))

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    raise ValueError(f"Unsupported file type: {ext}")


def get_dataframe_from_loaded_object(obj):
    if isinstance(obj, pd.DataFrame):
        return obj

    if isinstance(obj, dict):
        st.json(obj, expanded=False)
        return pd.DataFrame(obj)

    if isinstance(obj, list):
        return pd.DataFrame(obj)

    if isinstance(obj, str):
        st.text(obj[:5000])
        return None

    st.write(type(obj))
    st.write(obj)
    return None


def load_dataframe_to_session(data_file):
    data_file = str(data_file)

    if (
        st.session_state.get("ml_loaded_data_file") == data_file
        and "ml_df" in st.session_state
    ):
        return st.session_state["ml_df"]

    obj = load_file(data_file)

    if isinstance(obj, sqlite3.Connection):
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'",
            obj,
        )

        if tables.empty:
            temp_error("No tables found in this SQLite database.")
            return None

        table_name = st.selectbox(
            "Select table",
            tables["name"].tolist(),
            key="db_table_select",
        )

        df = pd.read_sql(f"SELECT * FROM {table_name}", obj)

    else:
        df = get_dataframe_from_loaded_object(obj)

    if df is None:
        return None

    st.session_state["ml_loaded_data_file"] = data_file
    st.session_state["ml_df"] = df.copy()

    # Reset saved data config when a new data file is loaded
    st.session_state.pop("ml_data_config", None)

    return st.session_state["ml_df"]


# ============================================================
# Helpers
# ============================================================

def safe_eval_formula(formula, x):
    allowed = {
        "x": x,
        "np": np,
        "log10": np.log10,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "exp": np.exp,
        "where": np.where,
    }

    return eval(
        formula,
        {"__builtins__": {}},
        allowed,
    )


def show_xy_preview(df, smiles_col=None, x_col=None, y_col=None, class_col=None):
    preview_cols = []

    for col in [smiles_col, x_col, y_col, class_col]:
        if col and col in df.columns and col not in preview_cols:
            preview_cols.append(col)

    if st.button(
        "Show Input Review",
        use_container_width=True,
        type="primary",
        key="ml_show_input_review",
    ):
        if preview_cols:
            st.subheader("Model Input Preview")
            st.dataframe(
                df[preview_cols].head(10),
                use_container_width=True,
                hide_index=True,
            )
        else:
            temp_info("No valid input columns selected.")


def create_class_id_from_class_labels(
    df: pd.DataFrame,
    class_col: str,
    class_id_col: str,
    class_names: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:

    clean_class = df[class_col].astype("string").str.strip()

    class_mapping = {
        str(name).strip(): idx
        for idx, name in enumerate(class_names)
    }

    df[class_id_col] = clean_class.map(class_mapping)
    df.loc[df[class_col].isna(), class_id_col] = np.nan

    return df, class_mapping


def save_training_dataframe(
    df,
    workdir,
    task_type,
    structure_col,
    activity_col,
    target_col,
    prefix,
    output_format="pickle",
):
    workdir = Path(workdir)

    required_cols = [
        structure_col,
        activity_col,
        target_col,
    ]

    required_cols = [
        c for c in required_cols
        if c is not None and c in df.columns
    ]

    train_df = df[required_cols].copy()

    train_df = train_df.dropna(
        subset=[structure_col, target_col]
    )

    output_dir = workdir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "parquet":
        output_file = output_dir / f"{prefix}_{task_type}_data.parquet"

        train_df.to_parquet(
            output_file,
            index=False,
            compression="snappy",
        )

    elif output_format == "pickle":
        output_file = output_dir / f"{prefix}_{task_type}_data.pkl"

        payload = {
            "data": train_df,
            "structure_col": structure_col,
            "activity_col": activity_col,
            "target_col": target_col,
            "task_type": task_type,
        }

        with open(output_file, "wb") as f:
            pickle.dump(payload, f)

    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

    return output_file


# ============================================================
# Main design
# ============================================================

def design(data_file, workdir, task_type):
    task_type = str(task_type).lower()

    data = st.session_state.get("ml_data_config", None)
    features = st.session_state.get("ml_feature_config", {})

    df = load_dataframe_to_session(data_file)

    if df is None:
        return data, features

    st.subheader("Raw Dataset Preview")
    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns(2, vertical_alignment="bottom")

    with c1:
        selected_features = st.multiselect(
            "Features",
            MOL_REP_NAMES,
            default=features.get("features", ["ECFP4"]),
            key="ml_representations",
        )

    features["features"] = selected_features

    with c2:
        if any(fp in selected_features for fp in ["ECFP4", "ECFP6", "FCFP4", "FCFP6"]):
            fp_bits = st.selectbox(
                "Fingerprint Bits",
                [1024, 2048, 4096],
                index=[1024, 2048, 4096].index(features.get("fp_bits", 2048))
                if features.get("fp_bits", 2048) in [1024, 2048, 4096]
                else 1,
                key="ml_fp_bits",
            )
            features["fp_bits"] = fp_bits
        else:
            features["fp_bits"] = None

    st.session_state["ml_feature_config"] = features

    c3, c4 = st.columns(2, vertical_alignment="bottom")

    with c3:
        structure_col = st.selectbox(
            "Structure Column",
            df.columns.tolist(),
            key="ml_smiles_col",
        )

    with c4:
        activity_col = st.selectbox(
            "Activity Column",
            df.columns.tolist(),
            key="ml_target_col",
        )

    activity = pd.to_numeric(df[activity_col], errors="coerce")

    y_col = activity_col
    class_col = None
    class_id_col = None
    n_classes = None

    # ============================================================
    # Regression
    # ============================================================

    if task_type == "regression":
        c5, c6, c7 = st.columns(3, vertical_alignment="bottom")

        with c5:
            conversion_formula = st.text_input(
                "Target Conversion Formula",
                value="9 - log10(x)",
                help="Use x as the selected activity column. Example: 9 - log10(x) for nM to pActivity.",
                key="ml_target_formula",
            )

        with c6:
            y_col = st.text_input(
                "New Y Column Name",
                value="pIC50",
                key="ml_y_col",
            )

        with c7:
            if st.button(
                "Apply Target Conversion",
                key="ml_apply_conversion",
                use_container_width=True,
                type="primary",
            ):
                try:
                    df[y_col] = safe_eval_formula(conversion_formula, activity)
                    st.session_state["ml_df"] = df
                    st.session_state.pop("ml_data_config", None)
                    temp_success(f"Created regression target column: {y_col}")
                except Exception as e:
                    temp_error(f"Invalid conversion formula: {e}")

        show_xy_preview(
            df=df,
            smiles_col=structure_col,
            x_col=activity_col,
            y_col=y_col if y_col in df.columns else None,
        )

    # ============================================================
    # Classification
    # ============================================================

    elif task_type == "classification":
        c5, c6 = st.columns(2, vertical_alignment="bottom")

        with c5:
            classification_mode = st.selectbox(
                "Classification Mode",
                [
                    "Binary: active/inactive by threshold",
                    "Multiclass: bins by thresholds",
                ],
                key="ml_classification_mode",
            )

        with c6:
            use_converted_y = st.checkbox(
                "Convert activity before classification",
                value=True,
                key="ml_use_converted_y_for_classification",
            )

        if use_converted_y:
            c7, c8, c9 = st.columns(3, vertical_alignment="bottom")

            with c7:
                conversion_formula = st.text_input(
                    "Target Conversion Formula",
                    value="9 - log10(x)",
                    help="Use x as the selected activity column. Example: 9 - log10(x) for nM to pActivity.",
                    key="ml_classification_target_formula",
                )

            with c8:
                y_col = st.text_input(
                    "Converted Y Column Name",
                    value="pIC50",
                    key="ml_classification_y_col",
                )

            with c9:
                if st.button(
                    "Apply Conversion",
                    key="ml_apply_classification_conversion",
                    use_container_width=True,
                    type="primary",
                ):
                    try:
                        df[y_col] = safe_eval_formula(conversion_formula, activity)
                        st.session_state["ml_df"] = df
                        st.session_state.pop("ml_data_config", None)
                        temp_success(f"Created converted target column: {y_col}")
                    except Exception as e:
                        temp_error(f"Invalid conversion formula: {e}")

            if y_col in df.columns:
                y = pd.to_numeric(df[y_col], errors="coerce")
            else:
                y = activity

        else:
            y_col = activity_col
            y = activity

        class_col = st.text_input(
            "Class Column Name",
            value=f"{activity_col}_class",
            key="ml_class_col",
        )

        class_id_col = f"{class_col}_id"

        if classification_mode == "Binary: active/inactive by threshold":
            c10, c11, c12 = st.columns(3, vertical_alignment="bottom")

            with c10:
                threshold = st.number_input(
                    "Classification Threshold",
                    value=6.0,
                    key="ml_binary_threshold",
                )

            with c11:
                active_direction = st.radio(
                    "Active Direction",
                    [
                        "Higher value = more active",
                        "Lower value = more active",
                    ],
                    horizontal=True,
                    key="ml_active_direction",
                )

            class_names = ["inactive", "active"]
            n_classes = len(class_names)

            with c12:
                if st.button(
                    "Step 1: Create Binary Classes",
                    key="ml_create_binary_class",
                    use_container_width=True,
                    type="primary",
                ):
                    if active_direction == "Higher value = more active":
                        df[class_col] = np.where(y >= threshold, "active", "inactive")
                    else:
                        df[class_col] = np.where(y <= threshold, "active", "inactive")

                    df.loc[y.isna(), class_col] = np.nan

                    if class_id_col in df.columns:
                        df = df.drop(columns=[class_id_col])

                    st.session_state["ml_df"] = df
                    st.session_state.pop("ml_data_config", None)
                    temp_success(f"Created class label column: {class_col}")

        else:
            c10, c11, c12, c13 = st.columns(4, vertical_alignment="bottom")

            with c10:
                thresholds_text = st.text_input(
                    "Thresholds, comma-separated",
                    value="5,6,7",
                    key="ml_multiclass_thresholds",
                )

            with c11:
                class_names_text = st.text_input(
                    "Class Names, comma-separated",
                    value="inactive,weak,moderate,strong",
                    key="ml_class_names",
                )

            class_names = [
                x.strip()
                for x in class_names_text.split(",")
                if x.strip()
            ]

            n_classes = len(class_names)

            with c12:
                st.metric("Number of classes", n_classes)

            with c13:
                if st.button(
                    "Step 1: Create Multiclass",
                    key="ml_create_multiclass",
                    use_container_width=True,
                    type="primary",
                ):
                    try:
                        thresholds = [
                            float(v.strip())
                            for v in thresholds_text.split(",")
                            if v.strip()
                        ]

                        class_names = [
                            v.strip()
                            for v in class_names_text.split(",")
                            if v.strip()
                        ]

                        if len(class_names) != len(thresholds) + 1:
                            temp_error(
                                "Number of class names must equal number of thresholds + 1."
                            )
                        else:
                            df[class_col] = pd.cut(
                                y,
                                bins=[-np.inf] + thresholds + [np.inf],
                                labels=class_names,
                            )

                            df.loc[y.isna(), class_col] = np.nan

                            if class_id_col in df.columns:
                                df = df.drop(columns=[class_id_col])

                            st.session_state["ml_df"] = df
                            st.session_state.pop("ml_data_config", None)
                            temp_success(f"Created class label column: {class_col}")

                    except Exception as e:
                        temp_error(f"Failed to create multiclass labels: {e}")

        st.markdown("#### Step 2: Map Classes to Digital IDs")

        if class_col not in df.columns:
            temp_info("Create the class label column first, then map it to digital IDs.")
        else:
            st.write("Class distribution:")
            st.write(df[class_col].value_counts(dropna=False))

            if classification_mode == "Binary: active/inactive by threshold":
                default_class_order = ["inactive", "active"]
            else:
                default_class_order = class_names

            class_order_text = st.text_input(
                "Class order for digital IDs",
                value=",".join(default_class_order),
                help="The first class becomes 0, second becomes 1, etc.",
                key="ml_class_order_text",
            )

            selected_class_order = [
                x.strip()
                for x in class_order_text.split(",")
                if x.strip()
            ]

            if st.button(
                "Step 2: Create Digital Class IDs",
                key="ml_create_class_ids",
                use_container_width=True,
                type="primary",
            ):
                existing_labels = (
                    df[class_col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )

                missing_labels = [
                    label for label in existing_labels
                    if label not in selected_class_order
                ]

                if missing_labels:
                    temp_error(
                        f"These labels exist in `{class_col}` but are missing from class order: "
                        f"{missing_labels}"
                    )
                else:
                    df, class_mapping = create_class_id_from_class_labels(
                        df=df,
                        class_col=class_col,
                        class_id_col=class_id_col,
                        class_names=selected_class_order,
                    )

                    if df[class_id_col].dropna().empty:
                        temp_error(f"No valid class IDs were created for `{class_id_col}`.")
                    else:
                        df[class_id_col] = pd.to_numeric(
                            df[class_id_col],
                            errors="coerce",
                        )

                        st.session_state["ml_df"] = df
                        st.session_state.pop("ml_data_config", None)

                        temp_success(f"Created digital class ID column: {class_id_col}")
                        st.write("Class mapping:")
                        st.json(class_mapping)

            if class_id_col in df.columns:
                st.write("Class ID distribution:")
                st.write(df[class_id_col].value_counts(dropna=False).sort_index())

                st.write("Class mapping preview:")
                st.dataframe(
                    df[[class_col, class_id_col]]
                    .drop_duplicates()
                    .sort_values(class_id_col),
                    use_container_width=True,
                    hide_index=True,
                )

        show_xy_preview(
            df=df,
            smiles_col=structure_col,
            x_col=activity_col,
            y_col=y_col,
            class_col=class_col if class_col and class_col in df.columns else None,
        )

    else:
        temp_error("task_type must be 'classification' or 'regression'.")
        return data, features

    # ============================================================
    # Save training dataframe
    # ============================================================

    target_col = y_col if task_type == "regression" else class_id_col

    st.markdown("#### Save Training Dataset")

    if target_col is None or target_col not in df.columns:
        temp_info("Please create/select a valid target column before saving training data.")
        return data, features

    valid_rows = df[[structure_col, target_col]].dropna().shape[0]

    st.write(f"Target column: `{target_col}`")
    st.write(f"Valid training rows: `{valid_rows}`")

    if valid_rows == 0:
        temp_error("No valid rows found after removing missing structure/target values.")
        return data, features

    if st.button(
        "Save Training Data",
        key="ml_save_training_data",
        use_container_width=True,
        type="primary",
    ):
        training_data_file = save_training_dataframe(
            df=df,
            workdir=workdir,
            task_type=task_type,
            structure_col=structure_col,
            activity_col=activity_col,
            target_col=target_col,
            prefix=Path(data_file).stem,
            output_format="pickle",
        )

        data = {
            "original_data_file": str(data_file) if data_file else None,
            "training_data_file": str(training_data_file),
            "X_col": structure_col,
            "y_col": target_col,
            "n_classes": n_classes,
        }

        st.session_state["ml_data_config"] = data
        st.session_state["ml_feature_config"] = features

        temp_success(f"Saved training data: {training_data_file}")

    else:
        if data is None:
            temp_info("Click **Save Training Data** when the setup is ready.")
        else:
            temp_success(f"Using saved training data: {data.get('training_data_file')}")

    return data, features